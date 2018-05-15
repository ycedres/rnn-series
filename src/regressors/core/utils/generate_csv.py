import os
import json

rootdir = os.path.dirname(os.path.realpath(__file__))
outputfile = open('output.csv','w')
header = "name,horizon,window_size,method,step_size,padding,batch_size,epochs,r2,mae,mse,rmse\n"
outputfile.write(header)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file=="description.json":
            # print(os.path.join(subdir,file))       
            with open(os.path.join(subdir,file),"rb") as f:
                data = json.load(f)
                
                line = data['name'] + ',' + \
                str(data['horizon']) + ',' + \
		        str(data['features_config']['window_size']) + ',' + \
		        str(data['features_config']['method']) + ',' + \
		        str(data['features_config']['step_size']) + ',' + \
		        str(data['features_config']['padding']) + ',' + \
		        str(data['train_config']['batch_size']) + ',' + \
		        str(data['train_config']['epochs']) + ',' + \
		        str(data['errors']['r2']) + ',' + \
		        str(data['errors']['mae']) + ',' + \
		        str(data['errors']['mse']) + ',' + \
		        str(data['errors']['rmse']) 
		        
                # print(line)
                outputfile.write(line+'\n')


		   
