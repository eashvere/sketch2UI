import sys
import numpy as np
import base64
import settings

def base64_encode_image(a):
    return base64.b64encode(a.tobytes()).decode('utf-8')

def base64_decode_image(a, dtype):
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

	# convert the string to a NumPy array using the supplied data
	# type and target shape
    a = np.frombuffer(base64.decodestring(a), settings.IMAGE_DTYPE)

    a = np.reshape(a, (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_CHANS))

    return a