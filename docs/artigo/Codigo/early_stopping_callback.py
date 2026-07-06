# Adicionando callback para parar o treino mais cedo caso o modelo pare de melhorar
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
