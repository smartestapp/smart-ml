import requests
import base64
import json
import sys
import os
import pandas as pd
from tqdm import tqdm
import time

SLEEP_DURATION = 5  # in seconds

# cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/oraquicktestpredict'
# cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/btnxtestpredict'
# cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/deepblueagtestpredict'
# cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/aconagtestpredict'
# cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/aconabtestpredict'
# cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/rapidconnectabtestpredict'
# cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/abbotttestpredict'
# cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/btnxtestpredictpredeploy'
# cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/quidelagtestpredict'
cloud_function_url = 'https://jivjy5pgt3.execute-api.us-east-2.amazonaws.com/covid/accessbioagtestpredict'

LABELS_PATH = None
PREDICTIONS_DIR = 'predictions'
OUTPUT_PATH = 'output.txt'

def main():
    """Main Method of Script"""
    if LABELS_PATH is not None:
        df = pd.read_excel(LABELS_PATH)
        df['statusCode'] = 0
        df['body'] = '<BODY>'
        df['Zone Pred'] = '[]'
        df['Diag Pred'] = '<>'
        df['Result'] = '...'
    else:
        df = pd.DataFrame({}, columns=['Sample ID', 'statusCode', 'body', 'inlet'])

    filenames = []

    for i, filename in enumerate(tqdm([line.strip() for line in open('batch_test_inputs.txt', 'r').readlines()], desc='Sending Predictions')):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG') or filename.endswith('.jpeg'): 
            filenames.append(filename)
            response = send_response({'image': filename, 'input_type': 'url'})

            response = response.json()
            print(filename, response)

            with open(OUTPUT_PATH, 'a+') as f:
                f.write('%s,%s\n' % (str(filename), str(response)) )

            if LABELS_PATH is not None:
                df.loc[df['Sample ID'] == filename.split('.')[0], 'statusCode'] = response['statusCode']

                # If we have a string ('...') instead of a list ([]), this means this is an error message!
                if isinstance(response['body'], str):
                    df.loc[df['Sample ID'] == filename.split('.')[0], 'body'] = response['body'] 
                    df.loc[df['Sample ID'] == filename.split('.')[0], 'Result'] = 'WRONG' 
                    continue

                df.loc[df['Sample ID'] == filename.split('.')[0], 'Zone Pred'] = str(response['body'][0])
                df.loc[df['Sample ID'] == filename.split('.')[0], 'Diag Pred'] = str(response['body'][1])

                zone_predictions = response['body'][0]
                if df[df['Sample ID'] == filename.split('.')[0]]['Zone 1'].values[0] != zone_predictions[0]:
                    df.loc[df['Sample ID'] == filename.split('.')[0], 'Result'] = 'WRONG' 
                elif df[df['Sample ID'] == filename.split('.')[0]]['Zone 2'].values[0] != zone_predictions[1]:
                    df.loc[df['Sample ID'] == filename.split('.')[0], 'Result'] = 'WRONG' 
                elif df[df['Sample ID'] == filename.split('.')[0]]['Zone 3'].values[0] != zone_predictions[2]:
                    df.loc[df['Sample ID'] == filename.split('.')[0], 'Result'] = 'WRONG' 
                else:
                    df.loc[df['Sample ID'] == filename.split('.')[0], 'Result'] = 'CORRECT' 
            else:
                out = [filename[:-4] if filename.endswith('.jpg') or filename.endswith('.png') else filename[:-5],
                       response['statusCode'], response['body']]
                if 'inlet' in out:
                    out.append(response['inlet'])
                else:
                    out.append('N/A')

                df.loc[i] = out

            time.sleep(SLEEP_DURATION)

    if LABELS_PATH is not None:
        df.to_csv('predictions_%s' % os.path.split(LABELS_PATH)[-1].replace('xlsx', 'csv'))
    else:
        df.to_csv(os.path.join(PREDICTIONS_DIR, '%s_predictions.csv' % os.path.split(PREDICTIONS_DIR)[-1]))

    with open(os.path.join(PREDICTIONS_DIR, 'filenames_correspondence.txt'), 'w') as f:
        for filename in filenames:
            f.write("%s\n" % filename)

    
def convert_to_base64string(image_path):
    """Converts the image passed in as a URL to base64string and returns it."""
    encoded_string = ''
    try:
        with open(image_path, 'rb') as image_file: #open the image path as an image_file
            encoded_string = base64.b64encode(image_file.read()) #encode the bytes read into base64
    except:
        print('Image path is not correct or the image can not be found.')
    
    base64string = encoded_string
    return base64string.decode('utf-8')

def send_response(image_dict):
    """Sends a response to the cloud function with payload = image_dict and return response from server."""
    image_json = json.dumps(image_dict)  # encodes the passed object into JSON format
    headers = {'content-type': 'application/json'}  # this header is required for the POST method
    
    response = ''
    try:
        # Send data through POST
        response = requests.post(cloud_function_url, data=image_json, headers=headers) 
    except:
        print('The request to the specified function could not be sent.')

    # Return the response object
    return response  
    
if __name__ == '__main__':
    main()
