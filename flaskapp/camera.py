import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIARRDLE7PMIA2OKXVV",
                        aws_secret_access_key="dekb3K1ssHEVZq50XY+1ODD1eXvC+8O9jl6VL00t",
                        aws_session_token="FwoGZXIvYXdzEFEaDKBy1j7NYfQxR+wULSLDAU8hoJRh3zI9sCTNSkns6bOC7DsP71OCGjKtQnIVNePaf7maMvgjRUnX6ykME9HYifg7j3vSYLD9AK0xraEnynD9LESb5rsuhkqONTtFxogleSLT9VXCr2P/KabxXKUboEfkMyvNs4e4NffvhXDKNlFDcAA96oC4b18zGDiHX3/cHTOpsMdNUnoPe7M32ZbfD+aU3rzofDit8/U3JueAts76upYrKv8bkAPfwbMoOJgl3RFrf1JScD5MCRrWYVjoA1KXlSin8OP6BTItimyVgDZXBTf3+q31LiShKuNbaTypS41l6//yAX41/rmJ6E6+cGKcy4GpaMc+",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:105451420632:project/MaskDetection/version/MaskDetection.2020-09-09T20.06.04/1599662165355',Image={
            'Bytes':image1})
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://rrx3ypocz1.execute-api.us-east-1.amazonaws.com/Main123?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
