# Dog Spotter ML

Projeto de Machine Learning para identificação e classificação de raças de cães utilizando Deep Learning com TensorFlow/Keras.

## 📋 Descrição

Este projeto implementa modelos de classificação de imagens para identificar raças de cães usando transfer learning com arquiteturas pré-treinadas. O sistema utiliza o dataset Stanford Dogs Dataset, que contém 120 raças diferentes de cães.

## 🏗️ Arquitetura

O projeto está organizado nas seguintes estruturas:

### Modelos Disponíveis

- **MobileNetV2**: Modelo leve e eficiente para classificação de raças de cães
- **ResNet50**: Modelo mais robusto baseado na arquitetura ResNet

### Estrutura do Projeto

```
ml/
├── source/
│   ├── app/                          # Aplicações de produção
│   │   ├── mobileNetV2/
│   │   │   └── src/
│   │   │       ├── model.py          # API Flask com MobileNetV2
│   │   │       ├── dog_spotter_model.keras
│   │   │       └── requirements.txt
│   │   └── resnet_dog_spotter/
│   │       └── src/
│   │           └── model.py          # API Flask com ResNet50
│   └── tests/
│       ├── examples/                 # Exemplos de uso
│       │   ├── mobileNetV2/
│       │   ├── resnet50/
│       │   ├── full_training/
│       │   ├── utils.py
│       │   └── requirements.txt
│       └── pocs/                     # Notebooks experimentais
│           ├── dogSpotter.ipynb
│           └── mobileNetV2_dogSpotter.ipynb
└── README.md
```

## 🚀 Tecnologias

- **TensorFlow 2.19.0**: Framework de Deep Learning
- **Keras**: API de alto nível para construção de redes neurais
- **Flask**: Framework web para servir o modelo via API REST
- **KaggleHub**: Download do dataset Stanford Dogs
- **NumPy**: Processamento numérico
- **Pillow**: Processamento de imagens
- **Matplotlib**: Visualização de dados

## 📦 Instalação

### Pré-requisitos

- Python 3.8+
- pip

### Instalação das dependências

```bash
# Clone o repositório
git clone <repository-url>
cd ml

# Crie um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instale as dependências do MobileNetV2
cd source/app/mobileNetV2/src
pip install -r requirements.txt

# OU instale as dependências dos exemplos
cd source/tests/examples
pip install -r requirements.txt
```

## 💻 Uso

### 1. Treinamento do Modelo MobileNetV2

```bash
cd source/tests/examples/mobileNetV2
python3 mobileNetV2.py
```

O script irá:
- Baixar automaticamente o Stanford Dogs Dataset
- Treinar o modelo MobileNetV2 por 50 épocas
- Salvar o modelo treinado em `dog_spotter_model.keras`

### 2. Executar API Flask

#### API MobileNetV2

```bash
cd source/app/mobileNetV2/src
python3 model.py
```

#### API ResNet50

```bash
cd source/app/resnet_dog_spotter/src
python3 model.py
```

A API estará disponível em `http://0.0.0.0:5000`

### 3. Fazer Predições via API

```bash
curl "http://localhost:5000/predict?id=1"
```

Resposta:
```json
{
  "request_id": 1,
  "result": "golden_retriever"
}
```

### 4. Teste com ResNet50 (Standalone)

```bash
cd source/tests/examples/resnet50
python3 resnet_dog_spotter.py
```

## 🧠 Modelos

### MobileNetV2

- **Arquitetura**: Transfer Learning com MobileNetV2 (ImageNet)
- **Input**: Imagens 240x240 pixels
- **Classes**: 120 raças de cães
- **Características**:
  - Base congelada (trainable=False)
  - GlobalAveragePooling2D
  - Dropout (0.2)
  - Dense layer com softmax
  - Mixed precision training (float16)
  - JIT compilation habilitada

### ResNet50

- **Arquitetura**: ResNet50 pré-treinada (ImageNet)
- **Input**: Imagens 224x224 pixels
- **Uso**: Inferência direta com pesos pré-treinados

## 📊 Dataset

**Stanford Dogs Dataset**
- 120 raças de cães
- Aproximadamente 20.580 imagens
- Fonte: Kaggle via KaggleHub
- Split: 80% treino / 20% validação

## 🔧 Configurações

### Hiperparâmetros (MobileNetV2)

```python
image_height = 240
image_width = 240
batch_size = 32
epochs = 50
learning_rate = 0.001
dropout_rate = 0.2
```

### Mixed Precision

O projeto utiliza mixed precision training para melhor performance:
```python
mixed_precision.set_global_policy('mixed_float16')
```

## 📝 API Endpoints

### GET /predict

Realiza predição de raça de cão a partir de uma imagem.

**Parâmetros:**
- `id` (int, required): ID da requisição

**Resposta de Sucesso (200):**
```json
{
  "request_id": 1,
  "result": "beagle"
}
```

**Resposta de Erro (400):**
```json
{
  "error": "Request ID is required"
}
```

**Resposta de Erro (500):**
```json
{
  "error": "Error message"
}
```

## 🔄 Pipeline de Treinamento

1. **Download do Dataset**: Via KaggleHub
2. **Preprocessamento**: Rescaling e data augmentation
3. **Transfer Learning**: Uso de pesos pré-treinados
4. **Fine-tuning**: Treinamento das camadas superiores
5. **Validação**: Split de 20% para validação
6. **Checkpoint**: Salvamento automático do modelo

## 📈 Performance

O modelo utiliza:
- **Data prefetching** para otimização de I/O
- **Shuffling** com buffer de 200 amostras
- **JIT compilation** para melhor performance
- **Mixed precision** para redução de memória

## 🐛 Troubleshooting

### Modelo corrompido

O sistema possui recuperação automática:
```python
# Se o modelo falhar ao carregar, ele é renomeado para .broken
# e um novo treinamento é iniciado automaticamente
```

### GPU não detectada

Verifique a instalação do TensorFlow GPU:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto faz parte do programa acadêmico da FATEC.

## 👥 Autores

Desenvolvido como parte do projeto Dog Finder.

## 🙏 Agradecimentos

- Stanford Dogs Dataset
- TensorFlow/Keras community
- FATEC

---

**Branch atual:** `feature@mobineNetV2_model`
