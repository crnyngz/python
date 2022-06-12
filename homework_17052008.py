import cv2
import numpy as np
import math

#Function to calculate PSNR in decibel units
def calculate_PSNR(original, equalized):
    
    M, N = original.shape
    epsilon = 0

    for i in range(M):
        for j in range(N):
            epsilon += (int(original[i][j]) - int(equalized[i][j])) ** 2

    epsilon /= (M * N)
    PSNR = 20 * np.log10(255 / np.sqrt(epsilon))

    return PSNR


def rgb2hsi(image):
    # HIS -> Hue Intensity Saturation
        blue, green, red = cv2.split(image)
        blue = blue.astype("uint32")
        green = green.astype("uint32")
        red = red.astype("uint32")
        # intensity calculation
        intensity = (blue + green + red) / 3
        intensity = intensity.astype("uint8")

        saturation = np.zeros(blue.shape)
        # saturation calculation
        for i in range(blue.shape[0]):
            for j in range(blue.shape[1]):
                if intensity[i][j] == 0:
                    saturation[i][j] = 0
                elif intensity[i][j] > 0:
                    minimum = np.minimum(np.minimum(red[i][j], green[i][j]), blue[i][j])
                    temp = red[i][j] + green[i][j] + blue[i][j]
                    saturation[i][j] = 1 - (3 * minimum / temp)

        hue = np.zeros(blue.shape)
        # hue calculation
        for i in range(blue.shape[0]):
            for j in range(blue.shape[1]):
                temp3 = red[i][j] ** 2 + green[i][j] ** 2 + blue[i][j] ** 2 - red[i][j] * blue[i][j] - red[i][j] * \
                        green[i][j] - green[i][j] * blue[i][j]
                if temp3 == 0:
                    temp3 += 0.000001
                temp2 = (red[i][j] - green[i][j] / 2 - blue[i][j] / 2) / math.sqrt(temp3)
                temp = math.acos(temp2)
                if green[i][j] >= blue[i][j]:
                    hue[i][j] = temp
                elif blue[i][j] > green[i][j]:
                    hue[i][j] = 360 - temp
                else:
                    pass

        return hue, saturation, intensity


def hsi2rgb(hue, saturation, intensity):

    red = np.zeros(hue.shape, dtype="uint8")
    green = np.zeros(hue.shape, dtype="uint8")
    blue = np.zeros(hue.shape, dtype="uint8")

    # true red color angle is 0
    # true green color angle is 120
    # true blue color angle is 240

    M, N = hue.shape
    for i in range(M):
        for j in range(N):
            if hue[i][j] == 0:
                red[i][j] = intensity[i][j] + 2 * intensity[i][j] * saturation[i][j]
                green[i][j] = intensity[i][j] - intensity[i][j] * saturation[i][j]
                blue[i][j] = intensity[i][j] - intensity[i][j] * saturation[i][j]
            elif hue[i][j] > 0 and hue[i][j] < 120:
                red[i][j] = intensity[i][j] + intensity[i][j] * saturation[i][j] * (np.cos(math.radians(hue[i][j])) /
                                                                                np.cos(math.radians(60 - hue[i][j])))
                green[i][j] = intensity[i][j] + intensity[i][j] * saturation[i][j] * ((1 - np.cos(math.radians(hue[i][j])))
                                                                                / np.cos(math.radians(60 - hue[i][j])))
                blue[i][j] = intensity[i][j] - intensity[i][j] * saturation[i][j]
            elif hue[i][j] == 120:
                red[i][j] = intensity[i][j] - intensity[i][j] * saturation[i][j]
                green[i][j] = intensity[i][j] + 2 * intensity[i][j] * saturation[i][j]
                blue[i][j] = intensity[i][j] - intensity[i][j] * saturation[i][j]
            elif hue[i][j] > 120 and hue[i][j] < 240:
                red[i][j] = intensity[i][j] - intensity[i][j] * saturation[i][j]
                green[i][j] = intensity[i][j] + intensity[i][j] * saturation[i][j] * (np.cos(math.radians(hue[i][j] -
                                                                        120)) / np.cos(math.radians(180 - hue[i][j])))
                blue[i][j] = intensity[i][j] + intensity[i][j] * saturation[i][j] * ((1 - np.cos(math.radians(hue[i][j]
                                                                    - 120))) / np.cos(math.radians(180 - hue[i][j])))
            elif hue[i][j] == 240:
                red[i][j] = intensity[i][j] - intensity[i][j] * saturation[i][j]
                green[i][j] = intensity[i][j] - intensity[i][j] * saturation[i][j]
                blue[i][j] = intensity[i][j] + 2 * intensity[i][j] * saturation[i][j]
            elif hue[i][j] > 240 and hue[i][j] < 360:
                red[i][j] = intensity[i][j] + intensity[i][j] * saturation[i][j] * ((1 - np.cos(math.radians(hue[i][j]
                                                                     - 240))) / np.cos(math.radians(300 - hue[i][j])))
                green[i][j] = intensity[i][j] - intensity[i][j] * saturation[i][j]
                blue[i][j] = intensity[i][j] + intensity[i][j] * saturation[i][j] * (np.cos(math.radians(hue[i][j] -
                                                                     240)) / np.cos(math.radians(300 - hue[i][j])))
    rgb = cv2.merge((blue, green, red))
    return rgb


# Read the image file
image = cv2.imread("mandrill.ppm", cv2.IMREAD_COLOR)

# Convert the color space to YCrCb and OpenCV default color space is BGR, not RGB
image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Split the channels for processing
channels_Y, channels_Cr, channels_Cb = cv2.split(image_YCrCb)

# Convert the color space to HSI and split channels
channels_H, channels_S, channels_I = rgb2hsi(image)

# Equalize histograms for the intensity channel on both color spaces
channels_Y_equ = cv2.equalizeHist(channels_Y)
channels_I_equ = cv2.equalizeHist(channels_I)

# Merge the channels back to reconstruct the image
equ_image_YCrCb = cv2.merge((channels_Y_equ, channels_Cr, channels_Cb))

# Convert the image back to RGB space
# equ_image_HSV = cv2.cvtColor(equ_image_HSV, cv2.COLOR_HSV2BGR)
equ_image_HSI = hsi2rgb(channels_H, channels_S, channels_I_equ)
equ_image_YCrCb = cv2.cvtColor(equ_image_YCrCb, cv2.COLOR_YCrCb2BGR)

# Calculate peak signal-to-noise ratio for each equalization
PSNR_HSI = calculate_PSNR(channels_I, channels_I_equ)
PSNR_YCrCb = calculate_PSNR(channels_Y, channels_Y_equ)

print(f"PSNR for HSV histogram equalization: {PSNR_HSI:.2f} dB")
print(f"PSNR for YCrCb histogram equalization: {PSNR_YCrCb:.2f} dB")

# Show results
cv2.imshow("image", image)
cv2.imshow("HSI", equ_image_HSI)
cv2.imshow("YCrCb", equ_image_YCrCb)

# Save file to results
cv2.imwrite("HSI_result.ppm", equ_image_HSI)
cv2.imwrite("YCrCb_result.ppm", equ_image_YCrCb)

cv2.waitKey(0)
cv2.destroyAllWindows()