import os
import sys
import time
import logging
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

# 设置环境变量避免Qt警告
os.environ['QT_QPA_PLATFORM'] = 'xcb'

class ANICalculator:
    def __init__(self):
        self.setup_logging()
        self.supported_ext = ['.fa', '.fasta', '.fna', '.ffn']
        
    def setup_logging(self):
        """配置日志记录"""
        self.logger = logging.getLogger('ANI_CALC')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        
        # 文件输出（稍后设置路径）
        self.fh = None
        
        self.logger.addHandler(ch)
    
    def init_file_logging(self, output_dir):
        """初始化文件日志"""
        if self.fh:
            self.logger.removeHandler(self.fh)
        
        log_file = os.path.join(output_dir, 'ani_calculation.log')
        self.fh = logging.FileHandler(log_file)
        self.fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.fh)
    
    def get_input_directory(self):
        """交互式获取输入目录"""
        while True:
            input_dir = input("请输入包含基因组文件的目录路径（或留空使用当前目录）: ").strip()
            if not input_dir:
                input_dir = os.getcwd()
            
            if os.path.isdir(input_dir):
                return input_dir
            print(f"错误: 目录 '{input_dir}' 不存在，请重新输入")
    
    def get_output_directory(self):
        """交互式获取输出目录"""
        while True:
            output_dir = input("请输入结果输出目录路径（或留空使用当前目录）: ").strip()
            if not output_dir:
                output_dir = os.getcwd()
            
            try:
                os.makedirs(output_dir, exist_ok=True)
                return output_dir
            except Exception as e:
                print(f"错误: 无法创建输出目录 '{output_dir}': {str(e)}")
    
    def get_genome_files(self, input_dir):
        """获取目录下的基因组文件"""
        genome_files = []
        for ext in self.supported_ext:
            genome_files.extend(list(Path(input_dir).glob(f'*{ext}')))
        
        if not genome_files:
            raise FileNotFoundError(
                f"未在 {input_dir} 中找到支持的基因组文件\n"
                f"支持的扩展名: {', '.join(self.supported_ext)}"
            )
        
        return sorted(genome_files)
    
    def validate_pyorthoani(self):
        """验证pyorthoani是否可用"""
        try:
            result = subprocess.run(
                ['pyorthoani', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("pyorthoani未正确安装")
            return result.stdout.strip()
        except Exception as e:
            raise RuntimeError(f"无法找到pyorthoani: {str(e)}")
    
    def run_pyorthoani(self, genome1, genome2):
        """运行pyorthoani并返回原始结果"""
        cmd = ['pyorthoani', '-q', str(genome1), '-r', str(genome2)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
                check=True
            )
            
            self.logger.debug(f"命令: {' '.join(cmd)}")
            self.logger.debug(f"输出: {result.stdout.strip()}")
            
            # 直接返回原始数值，不做任何过滤（包括0值）
            return float(result.stdout.strip())
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("计算超时(10分钟)")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"命令执行失败: {e.stderr}")
        except ValueError as e:
            raise RuntimeError(f"输出解析失败: {str(e)}. 原始输出: {result.stdout}")
        except Exception as e:
            raise RuntimeError(f"未知错误: {str(e)}")
    
    def calculate_ani_matrix(self, genome_files):
        """计算ANI矩阵"""
        num_genomes = len(genome_files)
        ani_matrix = np.identity(num_genomes) * 100  # 对角线设为100%
        file_names = [f.name for f in genome_files]
        results = []
        
        self.logger.info(f"开始计算 {num_genomes} 个基因组之间的ANI...")
        
        with tqdm(total=num_genomes*(num_genomes-1)//2, 
                 desc="计算进度", 
                 unit="pair") as pbar:
            for i, j in combinations(range(num_genomes), 2):
                try:
                    ani = self.run_pyorthoani(genome_files[i], genome_files[j])
                    ani_matrix[i][j] = ani
                    ani_matrix[j][i] = ani
                    
                    results.append({
                        'Genome_A': file_names[i],
                        'Genome_B': file_names[j],
                        'ANI': ani,
                        'Similarity': f"{ani:.2f}%"
                    })
                except Exception as e:
                    self.logger.warning(f"计算 {file_names[i]} 和 {file_names[j]} 失败: {str(e)}")
                    ani_matrix[i][j] = np.nan
                    ani_matrix[j][i] = np.nan
                    results.append({
                        'Genome_A': file_names[i],
                        'Genome_B': file_names[j],
                        'ANI': np.nan,
                        'Similarity': str(e)
                    })
                finally:
                    pbar.update(1)
        
        return ani_matrix, results, file_names
    
    def save_results(self, results, ani_matrix, labels, output_dir):
        """保存所有结果文件"""
        # 成对结果
        df_pairs = pd.DataFrame(results)
        pairs_file = os.path.join(output_dir, 'ani_pairs.csv')
        df_pairs.to_csv(pairs_file, index=False)
        
        # ANI矩阵
        df_matrix = pd.DataFrame(ani_matrix, index=labels, columns=labels)
        matrix_file = os.path.join(output_dir, 'ani_matrix.csv')
        df_matrix.to_csv(matrix_file)
        
        # 统计信息
        stats = {
            'total_genomes': len(labels),
            'total_pairs': len(results),
            'success_pairs': sum(1 for r in results if not pd.isna(r['ANI'])),
            'failed_pairs': sum(1 for r in results if pd.isna(r['ANI'])),
            'min_ani': df_pairs['ANI'].min(),
            'max_ani': df_pairs['ANI'].max(),
            'mean_ani': df_pairs['ANI'].mean()
        }
        
        stats_file = os.path.join(output_dir, 'ani_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("=== ANI计算统计 ===\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        
        return pairs_file, matrix_file, stats_file
    
    def plot_heatmap(self, ani_matrix, labels, output_dir):
        """绘制热图"""
        plt.figure(figsize=(12, 10))
        
        # 简化标签显示
        short_labels = [os.path.splitext(label)[0][:15] for label in labels]
        
        # 创建热图
        ax = sns.heatmap(
            ani_matrix,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
            xticklabels=short_labels,
            yticklabels=short_labels,
            annot_kws={"size": 6},
            linewidths=0.2,
            mask=np.isnan(ani_matrix),
            cbar_kws={'label': 'ANI (%)'}
        )
        
        plt.title('Average Nucleotide Identity (ANI) Heatmap', pad=20, fontsize=14)
        plt.xlabel('Genomes', fontsize=12)
        plt.ylabel('Genomes', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        # 保存图像
        heatmap_file = os.path.join(output_dir, 'ani_heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return heatmap_file
    
    def run(self):
        """主运行方法"""
        print("\n=== PyOrthoANI 基因组ANI计算工具 ===")
        print("版本: 1.1 | 支持交互式操作和智能错误处理\n")
        
        try:
            # 验证pyorthoani
            version = self.validate_pyorthoani()
            print(f"检测到 pyorthoani 版本: {version}")
            
            # 交互式获取路径
            input_dir = self.get_input_directory()
            output_dir = self.get_output_directory()
            self.init_file_logging(output_dir)
            
            # 获取基因组文件
            genome_files = self.get_genome_files(input_dir)
            print(f"\n找到 {len(genome_files)} 个基因组文件:")
            for i, f in enumerate(genome_files[:5], 1):
                print(f"{i}. {f.name}")
            if len(genome_files) > 5:
                print(f"...以及另外 {len(genome_files)-5} 个文件")
            
            # 确认开始计算
            confirm = input("\n是否开始计算ANI并生成热图？(y/n): ").strip().lower()
            if confirm != 'y':
                print("计算已取消")
                return
            
            # 计算ANI矩阵
            start_time = time.time()
            ani_matrix, results, labels = self.calculate_ani_matrix(genome_files)
            elapsed_time = time.time() - start_time
            
            # 保存结果和可视化
            pairs_file, matrix_file, stats_file = self.save_results(
                results, ani_matrix, labels, output_dir
            )
            heatmap_file = self.plot_heatmap(ani_matrix, labels, output_dir)
            
            # 读取统计信息
            with open(stats_file, 'r') as f:
                stats = f.read()
            
            # 输出摘要
            print("\n=== 计算完成 ===")
            print(stats)
            print(f"耗时: {elapsed_time:.2f} 秒")
            print(f"\n生成的文件:")
            print(f"- 成对ANI结果: {pairs_file}")
            print(f"- ANI矩阵: {matrix_file}")
            print(f"- ANI统计: {stats_file}")
            print(f"- ANI热图: {heatmap_file}")
            print(f"- 详细日志: {os.path.join(output_dir, 'ani_calculation.log')}")
            
        except Exception as e:
            self.logger.error(f"程序运行出错: {str(e)}", exc_info=True)
            print(f"\n错误: {str(e)}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    calculator = ANICalculator()
    calculator.run()
