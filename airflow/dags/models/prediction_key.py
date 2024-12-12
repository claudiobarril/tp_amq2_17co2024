import hashlib


class PredictionKey:
    def from_pipeline(self, arr):
        keys = []
        hashes = []
        for row in arr:
            row_str = ' '.join(map(lambda a: a[:15], map(str, row)))
            keys.append(row_str)
            hashes.append(hashlib.sha256(row_str.encode()).hexdigest())

        return keys, hashes

    def from_dataframe(self, df):
        keys = df.apply(lambda row: ' '.join(map(lambda a: a[:15], map(str, row))), axis=1)
        hashes = keys.apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

        return keys, hashes