def goldstein(phase, alpha, psize=32):
    import numpy as np

    def apply_pspec(data):
        # NaN is allowed value
        assert not(alpha < 0), f'Invalid parameter value {alpha} < 0'
        wgt = np.power(np.abs(data)**2, alpha / 2)
        data = wgt * data
        return data

    def make_wgt(nxp, nyp):
        # Create arrays of horizontal and vertical weights
        wx = 1.0 - np.abs(np.arange(nxp // 2) - (nxp / 2.0 - 1.0)) / (nxp / 2.0 - 1.0)
        wy = 1.0 - np.abs(np.arange(nyp // 2) - (nyp / 2.0 - 1.0)) / (nyp / 2.0 - 1.0)
        # Compute the outer product of wx and wy to create the top-left quadrant of the weight matrix
        quadrant = np.outer(wy, wx)
        # Create a full weight matrix by mirroring the quadrant along both axes
        wgt = np.block([[quadrant, np.flip(quadrant, axis=1)],
                        [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]])
        return wgt

    def patch_goldstein_filter(data, wgt, psize):
        """Apply the Goldstein adaptive filter to the given data.

        Args:
            data: 2D numpy array of complex values representing the data to be filtered.

        Returns:
            2D numpy array of filtered data.
        """
        # Calculate alpha
        data = np.fft.fft2(data, s=(psize,psize))
        data = apply_pspec(data)
        data = np.fft.ifft2(data, s=(psize,psize))
        return wgt * data

    def apply_goldstein_filter(data):
        # Create an empty array for the output
        out = np.zeros(data.shape, dtype=np.complex64)
        # ignore processing for empty chunks
        if np.all(np.isnan(data)):
            return data
        # Create the weight matrix
        wgt_matrix = make_wgt(psize, psize)
        # Iterate over windows of the data
        for i in range(0, data.shape[0] - psize, psize // 2):
            for j in range(0, data.shape[1] - psize, psize // 2):
                # Create proocessing windows
                data_window = data[i:i+psize, j:j+psize]
                wgt_window = wgt_matrix[:data_window.shape[0],:data_window.shape[1]]
                # Apply the filter to the window
                filtered_window = patch_goldstein_filter(data_window, wgt_window, psize)
                # Add the result to the output array
                slice_i = slice(i, min(i + psize, out.shape[0]))
                slice_j = slice(j, min(j + psize, out.shape[1]))
                out[slice_i, slice_j] += filtered_window[:slice_i.stop - slice_i.start, :slice_j.stop - slice_j.start]
        return out

    return apply_goldstein_filter(phase)
