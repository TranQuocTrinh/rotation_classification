from paddleocr import PaddleOCR, draw_ocr

PADDLE = PaddleOCR(use_angle_cls=True, lang='en')
def paddle_ocr(img_path):
    result = PADDLE.ocr(img_path, cls=True)
    texts = [x[1][0] for x in result]
    boxes = [x[0] for x in result]        # (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    
    # draw result
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    # im_show = draw_ocr(image, boxes, txts, scores)
    # im_show = Image.fromarray(im_show)
    
    # img_name = img_path.split('/')[-1]
    # name_result = './result/' + img_name.split('.')[0] + '_' + 'paddle.' + img_name.split('.')[-1]
    # im_show.save(name_result)
    return texts#, boxes, im_show

import os
filenames = os.listdir('/paddle/data_sample_from_back')
for f in filenames:
    img_path = "/paddle/data_sample_from_back/" + f
    print(paddle_ocr(img_path))
    break
