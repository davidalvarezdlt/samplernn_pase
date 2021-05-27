import re

lab_regex = r'([0-9]+) ([0-9]+) '
lab_regex += r'(.+)\^(.+)-(.+)\+(.+)=(.+)@(.+)_(.+)'
lab_regex += r'/A:(.+)_(.+)_(.+)'
lab_regex += r'/B:(.+)-(.+)-(.+)@(.+)-(.+)&(.+)-(.+)#(.+)-(.+)\$(.+)-(.+)!(.+)-(.+);(.+)-(.+)\|(.+)'
lab_regex += r'/C:(.+)\+(.+)\+(.+)'
lab_regex += r'/D:(.+)_(.+)'
lab_regex += r'/E:(.+)\+(.+)@(.+)\+(.+)&(.+)\+(.+)#(.+)\+(.+)'
lab_regex += r'/F:(.+)_(.+)'
lab_regex += r'/G:(.+)_(.+)'
lab_regex += r'/H:(.+)\=(.+)@(.+)=(.+)\|(.+)'
lab_regex += r'/I:(.+)=(.+)'
lab_regex += r'/J:(.+)\+(.+)-(.+)'


def read_lab(lab_file_path):
    lab_read_lines = []
    with open(lab_file_path) as lab_file:
        for lab_line in lab_file.readlines():
            lab_read_lines.append(re.search(lab_regex, lab_line).groups())
    return lab_read_lines


class SampleRNNQuantizer:
    LINEAR_QUANT = 0
    ULAW_QUANT = 1
    _EPSILON = 1e-2
    _EPSILONs = 1e-6
    _MU = 255.
    _LOG_MU1 = 5.5451774444795623
    q_type = None
    q_levels = None

    def __init__(self, q_type_ulaw, q_levels):
        self.q_type = self.ULAW_QUANT if q_type_ulaw else self.LINEAR_QUANT
        self.q_levels = q_levels

    def quantize_zero(self):
        return self.q_levels // 2

    def quantize(self, samples):
        return self.quantize_linear(samples) if self.q_type == self.LINEAR_QUANT else self.quantize_ulaw(samples)

    def dequantize(self, samples):
        return self.dequantize_linear(samples) if self.q_type == self.LINEAR_QUANT else self.dequantize_ulaw(samples)

    def quantize_linear(self, samples):
        samples = samples.clone()
        samples -= samples.min(dim=-1)[0].expand_as(samples)
        samples /= samples.max(dim=-1)[0].expand_as(samples)
        samples *= self.q_levels - self._EPSILON
        samples += self._EPSILON / 2
        return samples.long()

    def dequantize_linear(self, samples):
        return samples.float() / (self.q_levels / 2) - 1

    def quantize_ulaw(self, x, max_value=1.0):
        # Convert to uLaw
        x = x.sign() * ((self._MU / max_value) * x.abs() + 1.).log() / self._LOG_MU1
        # Midrise encoding
        y = 0.5 * (x + 1.0)
        y *= (self.q_levels - self._EPSILONs)
        return y.long()

    def dequantize_ulaw(self, y):
        # Midrise decoding
        y = y.float() * 2.0 / self.q_levels - 1.0
        # Inverse uLaw
        x = (y.abs() * self._LOG_MU1).exp() - 1
        x = x.sign() * x / self._MU
        return x
