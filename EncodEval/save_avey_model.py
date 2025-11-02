from avey.modeling_avey import AveyForMaskedLM
model = AveyForMaskedLM.from_pretrained("avey")
model = model.base_avey_model
model.save_pretrained("avey-model")
