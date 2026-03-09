_base_ = ['oneformer3d_1xb4_scannetpp_spconv.py']

model = dict(
    decoder=dict(
        type='ScanNetQueryDecoderSDPA',
        attn_query_chunk_size=128,
        mask_query_chunk_size=64),
    criterion=dict(
        inst_criterion=dict(
            matcher=dict(
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(
                        type='MaskBCECostChunked',
                        weight=1.0,
                        chunk_size=32768,
                        query_chunk_size=256,
                        point_chunk_size=32768),
                    dict(
                        type='MaskDiceCostChunked',
                        weight=1.0,
                        query_chunk_size=256,
                        point_chunk_size=32768)
                ],
                topk=1))))
