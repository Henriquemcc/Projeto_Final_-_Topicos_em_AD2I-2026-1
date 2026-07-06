# Criando o callback para salvar o melhor modelo
checkpoint_callback = ModelCheckpoint(
    filepath='modelo_resnet50_plantvillage.keras', # Nome do arquivo a ser salvo
    monitor='val_loss',                      # Monitora a perda da validacao
    save_best_only=True,                     # Salva apenas se melhorar (sobrescreve o anterior)
    save_weights_only=False,                 # Salva a arquitetura + pesos + otimizador
    mode='min',                              # 'min' porque queremos que a perda diminua
    verbose=1                                # Mostra no console quando salvar
)