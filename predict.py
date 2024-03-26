from utils.eval import EvalData
import cv2
import os

# img = cv2.imread('src/images/no_croped/12.14HTM52CP051_2_21.00.jpg')
# pred = EvalData(frame=img)
# pred.show_result()

# img_path = os.listdir('src/images/no_croped')

# for path in img_path:
#     predict = EvalData(f'no_croped/{path}')
#     predict.show_result()
#     print(predict.generated_text)

font = cv2.FONT_HERSHEY_SIMPLEX 
cap = cv2.VideoCapture('src/videos/test5.mp4')

if cap.isOpened() == False:
    print('Video Not Working')
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('src/results/prediction.mp4', fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            try:
                res = EvalData(frame = frame)
            except:
                continue
            
            try:    
                # cv2.circle(frame, (int(res.d[0]), int(res.d[1])), 10, (255,0,255), 1)
                if float(res.generated_text) > 9.9:
                    continue
                cv2.putText(frame, f"Prediction : {res.generated_text}",  (int(frame.shape[1]/2-100), int(frame.shape[0]/2-100)),  font, 1,  (255, 255, 255),  2,  cv2.LINE_4,)
                print(res.generated_text)
                cv2.imshow('Frame', frame)
                out.write(frame)
            except:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        else:
            break