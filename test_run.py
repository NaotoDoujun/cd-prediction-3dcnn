import os
import numpy as np
import trimesh
from src.preprocess import convert_stl_to_aligned_voxels

def test_pipeline():
    print("1. テスト用の3Dメッシュ（球体）をローカルで生成中...")
    # ネットからダウンロードせず、trimeshの機能で直径1.0の球体を生成します
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    
    # 本番の動きをシミュレートするため、わざと原点から大きくズラし、
    # 空中に浮かせた状態にしてからSTLに保存します（アライメントのテストのため）
    mesh.apply_translation([10.0, -5.0, 8.5])
    
    test_stl_path = "data/raw_stl/test_sphere.stl"
    mesh.export(test_stl_path)
    print(f"   -> 浮いた状態のテスト用STLを保存しました: {test_stl_path}")
    
    print("\n2. アライメント調整付きボクセル化を実行中（解像度: 32）...")
    resolution = 32
    voxel_matrix = convert_stl_to_aligned_voxels(test_stl_path, resolution=resolution)
    
    print(f"   -> 変換完了！ 配列形状: {voxel_matrix.shape}")
    print(f"   -> 物体が存在するボクセル数: {int(np.sum(voxel_matrix))}")
    
    # ---------------------------------------------------------
    # 3. アライメントの検証
    # ---------------------------------------------------------
    print("\n3. アライメントが意図通りか検証します:")
    
    z_indices, y_indices, x_indices = np.where(voxel_matrix > 0)
    
    # Z軸（上下）の検証：一番底面（インデックス0）に接地しているか？
    min_z = np.min(z_indices)
    print(f"   [Z軸 (上下)] 最下部のインデックス: {min_z} (期待値: 0 -> 底面に接地しているか)")
    
    # Y軸（左右）の検証：配列の中央（32の半分＝15.5付近）を中心に分布しているか？
    mean_y = np.mean(y_indices)
    print(f"   [Y軸 (左右)] 重心のインデックス: {mean_y:.2f} (期待値: 15.5付近 -> 左右中央にあるか)")
    
    # X軸（前後）の検証：配列の中央を中心に分布しているか？
    mean_x = np.mean(x_indices)
    print(f"   [X軸 (前後)] 重心のインデックス: {mean_x:.2f} (期待値: 15.5付近 -> 前後中央にあるか)")

    # ---------------------------------------------------------
    # 4. ターミナル上での簡易ビジュアライズ（断面の表示）
    # ---------------------------------------------------------
    print("\n4. ボクセルデータの中心断面（真横から見た図）を表示します:")
    mid_y = resolution // 2
    slice_side = voxel_matrix[:, mid_y, :]  # [Z, X] の2D断面
    
    for z in reversed(range(resolution)):
        row_chars = "".join(["■" if slice_side[z, x] > 0 else "  " for x in range(resolution)])
        if "■" in row_chars:
            print(f"Z={z:02d} | {row_chars}")
            
    # 保存テスト
    output_npy = "data/processed_voxels/test_sphere.npy"
    np.save(output_npy, voxel_matrix)
    print(f"\n5. ボクセルデータを保存しました: {output_npy}")

if __name__ == "__main__":
    os.makedirs("data/raw_stl", exist_ok=True)
    os.makedirs("data/processed_voxels", exist_ok=True)
    
    test_pipeline()