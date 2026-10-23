from pydub import AudioSegment
import tempfile
def to_wav(src):
    dst = tempfile.mktemp(suffix='.wav')
    AudioSegment.from_file(src).export(dst, format='wav')#
    return dst