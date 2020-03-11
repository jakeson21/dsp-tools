class SquareTimingRecovery():
    def __init__(self, sps, length):
        self.dt = 0
        self.sps = sps
        self.block_Size = length       
        self.buffer = np.zeros((0,), dtype=np.complex64)
        self.t0 = np.zeros((0,), dtype=np.double)
        self.t_x = np.zeros((0,), dtype=np.double)
        self.m = 0
        assert(self.sps >= 4)
            

    def step(self, x):
        if self.buffer.size != x.size + 2*self.sps:
            self.buffer = np.zeros((x.size+2*self.sps,), dtype=x.dtype)
            # Create time vector
            self.t0 = np.arange(self.buffer.size, dtype=np.double) - self.sps
            self.t0 = self.t0/self.sps
            self.t_x = np.arange(x.size, dtype=np.double)/self.sps
            
        # Move last symbol periods into front of buffer
        self.buffer[:2*self.sps] = self.buffer[-2*self.sps:]
        self.buffer[2*self.sps:] = x
        
        # Calculate residual offset
        t_err = self.getTrsEstimate(self.buffer)
        f = interpolate.interp1d(self.t0, self.buffer, copy=False, bounds_error=False, fill_value=0.)
        out = f(self.t_x + t_err)
        self.dt = t_err
        return out
        
    
    def getTrsEstimate(self, x):
        L = x.size/self.sps
        N = self.sps
        k = np.arange(self.m*L*N, (self.m+1)*L*N)
        val = -2.0*np.pi*k/N
        mag_squared = np.abs(x) ** 2.
        Xm = np.dot(mag_squared, np.exp(1j*val))
        t_estimate = (-0.5/np.pi)*np.angle(Xm)
        self.m += 1
        return t_estimate
        
        
def upsample(x, n):
    return np.hstack((np.expand_dims(x, axis=1), np.zeros((x.size,n-1)))).reshape((-1,))
    

def rcosdesign(beta, span, sps, name='normal'):
    delay = span * sps / 2
    t = np.arange(-delay, delay + 1.) / sps
    if name == 'normal':
        # Design a normal raised cosine filter
        b = np.zeros((t.size,))
        # Find non-zero denominator indices
        denom = 1. - (2. * beta * t) ** 2
        idx1 = np.abs(denom) > np.sqrt(sys.float_info.epsilon)

        # Calculate filter response for non-zero denominator indices
        b[idx1] = np.sinc(t[idx1]) * (np.cos(np.pi * beta * t[idx1]) / denom[idx1]) / sps

        # fill in the zeros denominator indices
        b[idx1 == False] = beta * np.sin(np.pi / (2 * beta)) / (2 * sps)
    else:
        # Design a square root raised cosine filter
        b = np.zeros((t.size,))
        # Find mid-point
        idx1 = t == 0
        if np.any(idx1):
            b[idx1] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)

        # Find non-zero denominator indices
        idx2 = np.abs(np.abs(4. * beta * t) - 1.0) < np.sqrt(sys.float_info.epsilon)
        if np.any(idx2):
            b[idx2] = 1 / (2 * np.pi * sps) \
                      * (np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta))
                         - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta))
                         + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta)))

        # fill in the zeros denominator indices
        nind = t[idx1 + idx2 == False]
        b[idx1 + idx2 == False] = (-4 * beta / sps) * (np.cos((1 + beta) * np.pi * nind) + np.sin((1 - beta) * np.pi * nind) / (4 * beta * nind)) / (np.pi * ((4 * beta * nind) ** 2 - 1))
    # Normalize filter energy
    b = b / np.sqrt(np.sum(b ** 2))
    return b
    
