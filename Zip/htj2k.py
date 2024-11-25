import json
import cv2
import os
import subprocess

def merge_roi_and_non_roi(image, roi_coords, output_file, htj2k_quality=10):
    """
    合并 ROI 和非 ROI 数据为单个文件。
    :param image: 输入图像。
    :param roi_coords: ROI 区域的坐标 (x, y, w, h)。
    :param output_file: 合并后的文件路径。
    :param htj2k_quality: HTJ2K 压缩质量。
    """
    x, y, w, h = roi_coords

    # 提取 ROI
    roi = image[y:y + h, x:x + w]
    _, roi_data = cv2.imencode('.png', roi)

    # 创建非 ROI 掩码
    mask = np.ones(image.shape[:2], dtype=np.uint8)
    mask[y:y + h, x:x + w] = 0

    # 提取非 ROI
    non_roi = cv2.bitwise_and(image, image, mask=mask)
    temp_non_roi_path = "temp_non_roi.png"
    cv2.imwrite(temp_non_roi_path, non_roi)

    # 使用 OpenJPEG 压缩非 ROI
    non_roi_output = "non_roi_compressed.j2k"
    command = [
        "opj_compress",
        "-i", temp_non_roi_path,
        "-o", non_roi_output,
        "-r", str(htj2k_quality)
    ]
    subprocess.run(command, check=True)

    # 读取 HTJ2K 数据
    with open(non_roi_output, "rb") as f:
        non_roi_data = f.read()

    # 删除临时文件
    os.remove(temp_non_roi_path)
    os.remove(non_roi_output)

    # 创建元数据
    metadata = {
        "roi_coords": roi_coords,
        "roi_size": len(roi_data),
        "non_roi_size": len(non_roi_data),
    }
    metadata_bytes = json.dumps(metadata).encode('utf-8')

    # 写入到单个文件
    with open(output_file, "wb") as f:
        f.write(len(metadata_bytes).to_bytes(4, 'big'))  # 写入元数据长度
        f.write(metadata_bytes)                         # 写入元数据
        f.write(roi_data)                               # 写入 ROI 数据
        f.write(non_roi_data)                           # 写入非 ROI 数据

    print(f"Combined file saved at {output_file}")


def extract_and_decode_combined_file(input_file):
    """
    从合并文件中分离 ROI 和非 ROI 数据。
    :param input_file: 合并文件路径。
    """
    with open(input_file, "rb") as f:
        metadata_size = int.from_bytes(f.read(4), 'big')  # 读取元数据长度
        metadata_bytes = f.read(metadata_size)           # 读取元数据
        metadata = json.loads(metadata_bytes.decode('utf-8'))

        roi_size = metadata["roi_size"]
        non_roi_size = metadata["non_roi_size"]
        roi_coords = metadata["roi_coords"]

        # 读取 ROI 数据
        roi_data = f.read(roi_size)
        with open("decoded_roi.png", "wb") as roi_file:
            roi_file.write(roi_data)

        # 读取非 ROI 数据
        non_roi_data = f.read(non_roi_size)
        with open("decoded_non_roi.j2k", "wb") as non_roi_file:
            non_roi_file.write(non_roi_data)

    print("ROI and Non-ROI data extracted successfully.")
    print(f"ROI saved as 'decoded_roi.png', Non-ROI saved as 'decoded_non_roi.j2k'.")


# 示例运行
if __name__ == "__main__":
    # 输入图像路径
    image_path = "input.jpg"
    image = cv2.imread(image_path)

    # 定义 ROI 区域
    roi_coords = (100, 100, 200, 200)  # 替换为实际 ROI 坐标 (x, y, w, h)

    # 合并输出文件
    combined_file = "combined_image.bin"
    merge_roi_and_non_roi(image, roi_coords, combined_file)

    # 提取并解码
    extract_and_decode_combined_file(combined_file)
