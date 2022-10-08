import cv2  
import numpy as np
import os
import face_recognition
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

mail_content = ''' Unauthorized user detected'''

cap = cv2.VideoCapture(0)   # 0 = main camera , 1 = extra connected webcam and so on.

neha_image = face_recognition.load_image_file("dataset\\dataset1.jpg")
neha_face_encoding = face_recognition.face_encodings(neha_image)[0]

raghav_image = face_recognition.load_image_file("dataset\\dataset2.jpg")
raghav_face_encoding = face_recognition.face_encodings(raghav_image)[0]

known_face_encodings = [neha_face_encoding,raghav_face_encoding]
known_face_names = ["Neha","Raghav"]

while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
            count = 0
            face_cascade = cv2.CascadeClassifier('E:\\Education\\Engineering\\Second Year\\4th Semester\\pr\\helo\\haarcascade_frontalface_default.xml')
            while(True):
                ret, img = cap.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                for (x,y,w,h) in faces:
                    color = (255, 0, 0)
                    stroke = 2
                    end_cord_x = x+w
                    end_cord_y = y+h
                    cv2.rectangle(img, (x,y), (end_cord_x,end_cord_y),color, stroke)     
                    count += 1
                        # Save the captured image into the datasets folder
                    cv2.imwrite("C:\\Users\\Admin\\Pictures\\unknown\\" + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    #cv2.imshow('image', img)
                k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= 1: # Take 1 face sample and stop video
                    break

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        

    # Display the resulting image
    cv2.imshow('Video', frame)
    if name=="Unknown":
        sender_address = 'patilneha0825@gmail.com'
        sender_pass = 'nehastar@marathe2508160118'
        receiver_address = 'patilneha0825@gmail.com'
            #Setup the MIME
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Subject'] = 'unauthorized user.'
                #The subject line
                #The body and the attachments for the mail
        message.attach(MIMEText(mail_content, 'plain'))
        attach_file_name = 'C:\\Users\\Admin\\Pictures\\unknown\\.1.jpg'
        attach_file = open(attach_file_name, 'rb') # Open the file as binary mode
        payload = MIMEBase('application', 'octate-stream')
        payload.set_payload((attach_file).read())
        encoders.encode_base64(payload) #encode the attachment
                #add payload header with filename
        payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
        message.attach(payload)
                #Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(sender_address, sender_pass) #login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
        print("Door Locked")
        print('Mail Sent')
    else:
        print("Door Unlocked")
    

    k = cv2.waitKey(5) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break


            
cap.release()
cv2.destroyAllWindows()
