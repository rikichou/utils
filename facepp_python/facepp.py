import urllib.request
import urllib.error
import time
import json

class FacePp(object):
    """
    Get face attribute use face ++ sdk
    """
    def __init__(self):
        self.http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
        self.key = "HoHrjI6XPacLAdDrh66Qv4KdKTbHRGEQ"
        self.secret = "eBpq-FUguZdOnhkKw78Z5pCP1K0eCKEj"

    def save_json(self, json_info, json_file_path):
        # save json
        with open(json_file_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(json_info, indent=2, ensure_ascii=False))

    def read_json(self, json_file_path):
        # read json
        with open(json_file_path) as f:
            t = json.load(f)
        return t

    def get_face_attr(self, image_path):
        """
        Get face attr
        :param image_path:  local path or url
        :return: json or None
        """
        boundary = '----------%s' % hex(int(time.time() * 1000))
        data = []
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
        data.append(self.key)
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
        data.append(self.secret)
        data.append('--%s' % boundary)
        fr = open(image_path, 'rb')
        data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
        data.append('Content-Type: %s\r\n' % 'application/octet-stream')
        data.append(fr.read())
        fr.close()
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
        data.append('1')
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
        data.append(
            "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
        data.append('--%s--\r\n' % boundary)

        for i, d in enumerate(data):
            if isinstance(d, str):
                data[i] = d.encode('utf-8')  # change to byte type

        http_body = b'\r\n'.join(data)

        # build http request
        req = urllib.request.Request(url=self.http_url, data=http_body)

        # header
        req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

        try:
            # post data to server
            resp = urllib.request.urlopen(req, timeout=5)
            # get response
            qrcont = resp.read()
            # if you want to load as json, you should decode first,
            # for example: json.loads(qrcont.decode('utf-8'))
            # print(qrcont.decode('utf-8'))

            dic = json.loads(qrcont.decode('utf-8'))

            return dic
        except urllib.error.HTTPError as e:
            print(e.read().decode('utf-8'))
            return None