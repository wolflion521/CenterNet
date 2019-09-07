import cv2
import numpy as np

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    # input shape is the diameter of gaussian circles,
    # here we cal radius, m and n are radius
    y, x = np.ogrid[-m:m+1,-n:n+1]
    # y is column vector, x is row vector
    # something like meshgrid

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # this is the 2D gaussian formula :e^(square_distance / 2*sigma^2)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # np.finfo(h.dtype).eps  this is epsilon, remember
    return h   #

def draw_gaussian(heatmap, center, radius, k=1, delte=6):
    # heatmap is an empty image, and we gonna call draw_gaussian 3 times
    # for topleft , bottomright ,center heat map
    # the input here center ,is note the center of bounding box, and it is just a point
    # for eg. when the heatmap is topleft heat map ,the center is the topleft corner coorinate of the detection.

    diameter = 2 * radius + 1 # this trick surprise me
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / delte)
    # the gaussian2D method return a heatmap, and it is symmetrically distributed
    # the smaller the sigmar parameter , the heatmap distribution is sharper.
    # and from the recommended configuration, we can see that the center keypoint
    # heatmap distribution is sharper than topleft and bottom right corner heatmap
    # gaussian shape is (radius*2,radius*2)

    x, y = center
    # center has 3 types : topleft, bottomright , boundingbox center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)

    top, bottom = min(y, radius), min(height - y, radius + 1)
    # say height,width = 128, 128 , and center= 30,60, radius = 16
    # then left = min(30,16) = 16
    #      right = min(128-30,16+1) = 17
    #      top = min(60,16) = 16
    #      bottom = min(128-60,16+1) = 17
    # so what is the purpose of this operation?
    # cal the dist between keypoint to heatmap boundaries,
    # if the keypoint is far enough from the boundaries, we build a normal distribution
    # top,bottom,left,right are all set to the guassian distribution radius
    # if the keypoint is too close to the heatmap boundaries,
    # the relevant distance , will cut at the boundaries, so the top | bottom | left | right will be small than radius
    # --------------------------------
    # -                              -
    # -                              -
    # -    _________                 -
    # -   |   + +   |                -
    # -   | +     + |                -
    # -   |+   *   +|                -
    # -   | +     + |                -
    # -   |___+_+___|                -
    # -                              -
    # -                              -
    # -                              -
    # -                              -
    # --------------------------------

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    # look above diagram, masked_heatmap is just the rectangle area
    # and if the keypoint is too close to the boundaries it may lose some cols or rows
    # so following line keep it to the size of (radius*2,radius*2)
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    # this above three lines: reference to the relevant area, and name that area as masked_heatmap
    # gaussian is shape of (radius*2,radius*2) and because of the influence of boundaries,masked_gaussian
    # could be (radius*2,radius*2) or smaller than that ,we just cut relevant area from gaussian
    # then feed the cut gaussian into the area we referenced.
    # and why we need a np.maximum operation? sometimes , one single image may contain several same class objects
    # and the heat map finally will have multiple peaks.


def gaussian_radius(det_size, min_overlap):
    # det_size is short for detection size, which
    # here det_size is height width for detection box in output Layers
    # min_overlap is 0.7 base on CenterNet-104
    # the radius is only influenced by the object bounding box size
    #             has nothing to do with output size
    height, width = det_size

    # say output_size = 128,128
    #       there is an object, whose size is height = 50, width = 80
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    # a1 = 1, b1 = 130,c1 = 50*80*0.3/1.3 = 923.076923,b1 ** 2 - 4 * a1 * c1 = 16900 - 3692.307692 = 13207.692308
    # sq1 = 114    ,   r1 = (130 + 114)/2 = 122
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    # a2 = 4, b2 = 260, c2 = 0.3*50*80 = 1200 , b^2 - 4ac = 67600-9600=5800
    # sq2 = 240.83, r2 = (260 + 240)/2 = 250

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    # a3 = 4*0.7 = 2.8, b3 = -2 * 0.7*130 = -182, c3 = -0.3*50*80 = -1200
    # b^2 - 4ac = 33124 + 13440 = 46564,sq3 = 215.78
    # r3 = (-182+215)/2 = 16.5
    return min(r1, r2, r3)#(122,250,16.5) so return 16.5

def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
    # say-- ----size = 320 border = 128
    #            i = 1--> 320 - 128//1 = 192    128//1=128
    #            then return 192
    # say ------size = 240 border = 128
    #            i = 1 -->240 - 128//1 = 112    128//1=128
    #            i = 2 -->240 - 128//2 = 176    128//2=64
    #            then return 64
    # say ------size = 50 border = 128
    #            i = 1--> 50 - 128//1  = -78    128//1=128
    #            i = 2--> 50 - 128//2  = -14    128//2=64
    #            i = 4--> 50 - 128//4  = 18     128//4=32
    #            i = 8--> 50 - 128//8  = 34     128//8=16
    #            then return 16
    # say -------size = 800, border = 128
    #            i = 1--> 800 - 128//1
    #            then return 128
    # say -------size = 600,border = 128
    #            then return 128



def random_crop(image, detections, random_scales, view_size, border=64):
    ####
    # This method is just crop in pixels
    # if you feed in a picture of 800 x 600 , your network_size is (511,511), random_scales = 0.7,border is 128
    #                  then you need a crop region of (358,358) from picture, we just crop this piece
    #                  and change the detections
    ####
    # image is a cv2.mat
    # detections is the bounding box for a single image
    # rand_scales = np.arange(0.6,1.4,0.1)
    # when training : view_size = input_size = [511,511]
    #                 border = 128
    view_height, view_width   = view_size
    # 511      ,   511
    image_height, image_width = image.shape[0:2]
    # say 320,240
    # say*2nd   800,600

    scale  = np.random.choice(random_scales)
    # select a value in np.arange(0.6,1.4,0.1)
    # say-- scale = 0.7

    height = int(view_height * scale)
    width  = int(view_width  * scale)
    # say-- scale = 0.7
    #       height = 511*0.7 = 358
    #       width  = 511*0.7 = 358


    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)
    # (358,358,3)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)
    # for default CenterNet-104.json settings border is 128
    # say image_height, image_width = 320,240
    # then w_border  = 128, h_border = 64

    # say*2nd   image_height, image_width = 800,600
    # then*2nd  w_border  = 128, h_border = 128

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)
    # say image_height, image_width = 320,240
    # then w_border  = 128, h_border = 64
    # ctx is in range(128,320-128)=(128,192)
    # cty is in range (64,240-64) = (64,176)
    # say ctx,cty = 150, 120

    # say*2nd   image_height, image_width = 800,600
    # then*2nd  w_border  = 128, h_border = 128
    # then*2nd  ctx is in range(128,800-128)=(128,672)
    # then*2nd  cty is in range (128,600-128) = (128,472)
    # say*2nd   ctx,cty = 150, 200

    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)
    # say-- scale = 0.7
    #       height = 511*0.7 = 358
    #       width  = 511*0.7 = 358
    #
    # ctx is in range(128,320-128)=(128,192)
    # cty is in range (64,240-64) = (64,176)
    # say ctx,cty = 150, 120
    #
    # so  x0,x1 = max(150-358//2,0),min(150+358//2,320)
    #           = 0,320
    #     y0,y1 = max(120-358//2,0),min(120+358//2,240)
    #           = 0,240

    # say*2nd   image_height, image_width = 800,600
    # then*2nd  w_border  = 128, h_border = 128
    # then*2nd  ctx is in range(128,800-128)=(128,672)
    # then*2nd  cty is in range (128,600-128) = (128,472)
    # say*2nd   ctx,cty = 150, 200
    # then*2nd  x0,x1 = max(150-358//2,0),min(150+358//2,800)
    #     #           = 0,150+179 = 0,329
    #     #     y0,y1 = max(200-358//2,0),min(200+358//2,600)
    #     #           = 200 - 179, 200 + 179 = 21,379

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty
    # left_w, right_w = 250 - 71, 429 - 250 = 179, 179
    # top_h, bottom_h = 180 - 1,  359 - 180 = 179, 179

    # say*2nd   image_height, image_width = 800,600
    # then*2nd  w_border  = 128, h_border = 128
    # then*2nd  ctx is in range(128,800-128)=(128,672)
    # then*2nd  cty is in range (128,600-128) = (128,472)
    # say*2nd   ctx,cty = 150, 200
    # then*2nd  x0,x1 = max(150-358//2,0),min(150+358//2,800)
    #     #           = 0,150+179 = 0,329
    #     #     y0,y1 = max(200-358//2,0),min(200+358//2,600)
    #     #           = 200 - 179, 200 + 179 = 21,379
    # then*2nd  left_w, right_w = 150 - 0, 329 - 150 = 150, 179
    #           top_h, bottom_h = 200 - 21,  379 - 200 = 179, 179

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    #                         = 179,179
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    #               (  179  -  179      ,   179  + 179)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    #               (  179  -  179      ,   179  + 179)

    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]
    #                                         [1:359, 71:429,:]

    # say*2nd   image_height, image_width = 800,600
    # then*2nd  w_border  = 128, h_border = 128
    # then*2nd  ctx is in range(128,800-128)=(128,672)
    # then*2nd  cty is in range (128,600-128) = (128,472)
    # say*2nd   ctx,cty = 150, 200
    # then*2nd  x0,x1 = max(150-358//2,0),min(150+358//2,800)
    #     #           = 0,150+179 = 0,329
    #     #     y0,y1 = max(200-358//2,0),min(200+358//2,600)
    #     #           = 200 - 179, 200 + 179 = 21,379
    # then*2nd  left_w, right_w = 150 - 0, 329 - 150 = 150, 179
    #           top_h, bottom_h = 200 - 21,  379 - 200 = 179, 179
    #           cropped_ctx, cropped_cty = width // 2, height // 2 = 179,179
    #           x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)=slice(179 - 150,179+179) = slice(29,384)
    #           y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)=slice(179 - 179,179+179) = slice(0,384)
    #           cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]
    #           <==>cropped_image[slice(0,384), slice(29,384), :] = image[21:379, 0:150, :]
    #           until now the cropped image doesn't have any distortion in vertical/horizontal rate

    # crop detections
    cropped_detections = detections.copy()
    cropped_detections[:, 0:4:2] -= x0
    # x0 = 71
    # 0:4:2 ==  0,1
    # x coordinate move left x0
    cropped_detections[:, 1:4:2] -= y0
    # y coorinate move up y0
    cropped_detections[:, 0:4:2] += cropped_ctx - left_w
    # cropped_ctx = 179, left_w = 179
    #
    cropped_detections[:, 1:4:2] += cropped_cty - top_h
    # cropped_cty = 179, top_h  = 179

    return cropped_image, cropped_detections
