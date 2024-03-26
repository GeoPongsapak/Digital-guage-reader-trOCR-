from os import getcwd
import sys
sys.path.append(getcwd())
from config.libaries import *
from config.configuration import GENERAL_CONFIG

class EvalData:
    def __init__(self, file_name=None, frame=None) -> None:
        
        self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # self.image = Image.open(join(GENERAL_CONFIG.SOURCE_FOLDER, GENERAL_CONFIG.IMAGES_FOLDER + file_name))
        self.yolo_model = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER, GENERAL_CONFIG.MODEL_FOLDER+'YOLO/best_YOLO_number_1.pt'))
        self.crop_image()
    
    def crop_image(self):
        r = self.yolo_model(self.image, conf=0.3)
        for rt in r:
            orig_img = Image.fromarray(rt.orig_img[..., ::-1]).convert('RGB')
            name = rt.names
            names = []
            for c in rt.boxes.cls:
                names.append(name[int(c)])

            df = pd.DataFrame(np.zeros([len(names), 4]))
            df.columns = ['xmin', 'ymin', 'xmax', 'ymax']

            
            boxes = rt.boxes  
            for idx, box in enumerate(boxes):
                coordinate = box.xyxy
                row = np.array(coordinate, dtype=float)
                df.loc[idx] = row

            df['predict'] = names
            num_box = df[df['predict'] == 'number']
            # print(num_box)
            df = df.sort_values('xmin')
            # im_array = rt.plot() 
            # im = Image.fromarray(im_array[..., ::-1]).convert('RGB')

            x1 = num_box['xmin'].values.tolist()[0]
            y1 = num_box['ymin'].values.tolist()[0]
            y2 = num_box['ymax'].values.tolist()[0]
            x2 = num_box['xmax'].values.tolist()[0]
            
            self.coor_print = [x2,y2]
            self.im_crop = orig_img.crop((x1,y1,x2,y2))

            self.predict_number()

    def predict_number(self):
        print(torch.__version__)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        processor = TrOCRProcessor.from_pretrained(GENERAL_CONFIG.PROCESSOR_MODEL)
        trained_model = VisionEncoderDecoderModel.from_pretrained(join(GENERAL_CONFIG.SOURCE_FOLDER, GENERAL_CONFIG.MODEL_FOLDER + 'pt_add_edited_dataset_2_best')).to(device)

        pixel_values = processor(self.im_crop, return_tensors='pt').pixel_values.to(device)
        generated_ids = trained_model.generate(pixel_values)
        self.generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.final_text = self.generated_text[0] + "9" + self.generated_text[1:]


    def show_result(self):
        plt.imshow(self.image)
        plt.title(f"Result = {self.generated_text}")
        plt.axis(False)
        plt.show()