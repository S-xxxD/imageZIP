import cv2
import numpy as np
import subprocess
import os

def compress_image_htj2k(image, output_path, rate=20):
    """
    使用 HTJ2K 对图像进行压缩
    Args:
        image: 输入图像 (BGR 格式)。
        output_path: 输出压缩文件路径 (如 .j2k)。
        rate: 压缩码率，值越小压缩越强。
    """
    # 保存原始图像为临时文件 (PGM 或 PPM 格式是 JPEG 2000 支持的输入格式)
    temp_input = "temp_input.ppm"
    cv2.imwrite(temp_input, image)

    # 使用 OpenJPEG 的 opj_compress 进行 HTJ2K 压缩
    compress_command = [
        "opj_compress",  # OpenJPEG 命令行工具
        "-i", temp_input,  # 输入图像
        "-o", output_path,  # 输出 HTJ2K 文件
        "-r", str(rate)  # 压缩码率
    ]

    try:
        subprocess.run(compress_command, check=True)
        print(f"Compressed image saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error during HTJ2K compression:", e)
    finally:
        # 删除临时文件
        if os.path.exists(temp_input):
            os.remove(temp_input)


def decompress_image_htj2k(input_path):
    """
    使用 HTJ2K 对压缩文件解压缩回图像。
    Args:
        input_path: 输入的压缩文件路径 (.j2k)。
    Returns:
        解压缩后的图像。
    """
    # 输出的临时解压文件
    temp_output = "temp_output.ppm"

    # 使用 OpenJPEG 的 opj_decompress 解压
    decompress_command = [
        "opj_decompress",  # OpenJPEG 解压工具
        "-i", input_path,  # 输入压缩文件
        "-o", temp_output  # 输出解压图像
    ]

    try:
        subprocess.run(decompress_command, check=True)
        print(f"Decompressed image saved to {temp_output}")
        # 读取解压后的图像
        decompressed_image = cv2.imread(temp_output)
        return decompressed_image
    except subprocess.CalledProcessError as e:
        print("Error during HTJ2K decompression:", e)
        return None
    finally:
        # 删除临时文件
        if os.path.exists(temp_output):
            os.remove(temp_output)


# 示例调用
def main(image_path):
    # 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image!")
        return

    # HTJ2K 压缩
    compressed_path = "compressed_image.j2k"
    compress_image_htj2k(image, compressed_path, rate=10)  # 压缩码率为10

    # 解压缩
    decompressed_image = decompress_image_htj2k(compressed_path)
    if decompressed_image is not None:
        # 显示解压后的图像
        cv2.imshow("Decompressed Image", decompressed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "D:/PycharmProjects/Match/imageZIP/image/CO2.jpg"  # 替换为你的图像路径
    main(image_path)
