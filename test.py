from PIL import Image


def create_white_image(width, height):
    # 创建一个白色图片
    image = Image.new("RGB", (width, height), "white")

    # 保存图片
    image.save("white_image.png")


# 指定自定义分辨率大小
custom_width = 1200
custom_height = 628

# 调用函数创建白色图片
create_white_image(custom_width, custom_height)

