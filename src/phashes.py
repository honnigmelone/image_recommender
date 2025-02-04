
import cv2
from scipy.spatial import distance
from matplotlib import pyplot as plt

def calculate_mean(pixels_list):
    mean = 0
    total_pixels = len(pixels_list)
    for i in range(total_pixels):
        mean += pixels_list[i] / total_pixels
    return mean


def grab_pixels(squeezed_frame):
    pixels_list = []
    for x in range(0, squeezed_frame.shape[1], 1):
        for y in range(0, squeezed_frame.shape[0], 1):
            pixel_color = squeezed_frame[x, y]
            pixels_list.append(pixel_color)
    return pixels_list

def hashify(squeezed_frame, bits_list):
    bit_index = 0
    hashed_frame = squeezed_frame
    for x in range(0, squeezed_frame.shape[1], 1):
        for y in range(0, squeezed_frame.shape[0], 1):
            hashed_frame[x, y] = bits_list[bit_index]
            bit_index += 1
    return hashed_frame

def make_bits_list(mean, pixels_list):
    bits_list = []
    for i in range(len(pixels_list)):
        if pixels_list[i] >= mean:
            bits_list.append(255)
        else:
            bits_list.append(0)
    return bits_list

def generate_hash(frame, hash_size = 16):
    frame_squeezed = cv2.resize(frame, (hash_size, hash_size))
    frame_squeezed = cv2.cvtColor(frame_squeezed, cv2.COLOR_BGR2GRAY)
    pixels_list = grab_pixels(frame_squeezed)
    mean_color = calculate_mean(pixels_list)
    bits_list = make_bits_list(mean_color, pixels_list)
    hashed_frame = hashify(frame_squeezed, bits_list)
    hashed_frame = cv2.cvtColor(hashed_frame, cv2.COLOR_GRAY2BGR)
    return bits_list, hashed_frame


if __name__ == "__main__":
    img = cv2.imread(r"C:\Uni\SOSE_2024\big_data\project\experiments\1842.jpg")#"D:\data\image_data\Landscapes\00000021_(6).jpg"
    img_tocompare = cv2.imread(r"C:\Uni\SOSE_2024\big_data\project\experiments\000000414738.jpg")

    bits_list, hashed_frame = generate_hash(img)
    bits_list2, hashed_frame2 = generate_hash(img_tocompare)

    print(distance.hamming(bits_list, bits_list2))
    #print(bits_list)

    print(bits_list)
    print(hashed_frame)

    fig, (ax1, ax2) = plt.subplots(1, 2)  # 1 row, 2 columns

    # Display the first image in the first subplot
    ax1.imshow(hashed_frame)
    ax1.axis('off')  # Optional: turns off the axis

    # Display the second image in the second subplot
    ax2.imshow(hashed_frame2)
    ax2.axis('off')  # Optional: turns off the axis

    # Show the plot
    plt.show()
    """
    plt.imshow(hashed_frame)
    plt.imshow(hashed_frame2)
    """