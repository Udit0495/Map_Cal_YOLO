import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import json
import time

# cuda.init()
# device = cuda.Device(0)
# cuda_driver_context = device.make_context()


logger = trt.Logger(trt.Logger.WARNING)
logger.min_severity = trt.Logger.Severity.ERROR
runtime = trt.Runtime(logger)
trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins

# results = []
# results_file = 'result_file_final.json'

with open('yolov8s_trt.engine', "rb") as f:
    serialized_engine = f.read()
engine = runtime.deserialize_cuda_engine(serialized_engine)
imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
context = engine.create_execution_context()


def inference(img_path, engine, results=None):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    t1 = time.time()
    img = cv2.imread(img_path)
    #print(type(img))
    # hi, wi = img.shape[0], img.shape[1]
    hi, wi = 640, 640
    img = cv2.resize(img, (640,640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.transpose((2,0,1)).astype(np.float32))/255.0
    img = np.expand_dims(img, axis=0)

    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    inputs[0]['host'] = np.ravel(img)
    # transfer data to the gpu
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
    # run inference
    context.execute_async_v2(bindings=bindings,
                            stream_handle=stream.handle)
    # fetch outputs from gpu
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    # synchronize stream
    stream.synchronize()

    data = [out['host'] for out in outputs]
    
    outs = np.array(data[0])
    
    pred = (outs.reshape(-1,8400)).transpose()
    #pred = prediction.transpose()

    #print(pred.shape)

    boxes = pred[:,:4]
    score = np.amax(pred[:,4:], axis=1)
    
    nms = cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), 0.3, 0.5)
    print('time : ', time.time()-t1)
    #print(nms)

    #image_id = int(jpg.split('.')[0].split('/')[-1])
    # image = cv2.imread(img_path)
    # image = cv2.resize(image, (640,640))
    
    
    for i in nms:
        x, y, w, h = pred[i[0],:4]
        x = x-(w/2.0)
        y = y-(h/2.0)
        
        x, y, w, h = float(x), float(y), float(w), float(h)

        scale_h = hi/640.0
        scale_w = wi/640.0
        x = x*scale_w
        y = y*scale_h
        w = w*scale_w
        h = h*scale_h

        conf = np.max(pred[i[0],4:])
        cls = int(np.argmax(pred[i[0],4:]))

        # results.append({'image_id': image_id,
        #                     'category_id': cls,
        #                     'bbox': [x, y, w, h],
        #                     'score': float(conf)})
    
    del inputs
    del outputs
    del stream
    
#print(boxes)

# print(pred[idx][:4])
# img1 = cv2.imread('image1.webp')
# img1 = cv2.resize(img1, (640, 640))

# x1, y1, x2, y2 = pred[idx][:4]
# x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# x1 = x1-x2//2
# y1 = y1-y2//2
# x2 = x1+x2
# y2 = y1+y2

# img_v = cv2.rectangle(img1, (x1,y1), (x2,y2), (255,0,0), 2)
# plt.imshow(img_v)
# plt.show()

#jpgs = [j for j in os.listdir('images/') if j.endswith('.jpg')]

#i = 1
for j in range(1,101):
    print('image : ',j)
    #cuda_driver_context.push()
    inference('image2.jpg', engine)
    #cuda_driver_context.pop()
    #i+=1

# with open(results_file, 'w') as f:
#     f.write(json.dumps(results, indent=4))

# print('avg time : ', time.time()-t1)
