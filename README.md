Futura-IA – Orientación Académica FP con Inteligencia Artificial

Este proyecto consiste en una aplicación web de orientación académica basada en Inteligencia Artificial, orientada a la Formación Profesional (FP) de la Comunitat Valenciana.
La aplicación permite al alumnado responder a un cuestionario y recibir recomendaciones personalizadas de ciclos formativos, utilizando un modelo de lenguaje integrado a través de Amazon Bedrock.

Repositorio (rama utilizada):
👉 https://github.com/ArnauCatala/Futura-IA/tree/servidorDual

🚀 Puesta en marcha del proyecto
1. Requisitos previos

Antes de ejecutar la aplicación, es necesario disponer de:

Docker

Docker Compose

Una cuenta de AWS con acceso a Amazon Bedrock

Credenciales válidas de AWS

2. Clonar el repositorio
git clone https://github.com/ArnauCatala/Futura-IA.git
cd Futura-IA
git checkout servidorDual

3. Configuración del archivo .env

El proyecto utiliza variables de entorno para la conexión con Amazon Bedrock.
En el directorio raíz del proyecto, se debe crear o editar un archivo llamado .env con el siguiente contenido:

AWS_ACCESS_KEY_ID=TU_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=TU_SECRET_KEY
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=amazon.nova-pro-v1:0

Descripción de las variables:

AWS_ACCESS_KEY_ID: clave de acceso de la cuenta AWS.

AWS_SECRET_ACCESS_KEY: clave secreta asociada a la cuenta.

AWS_REGION: región donde está disponible Amazon Bedrock.

BEDROCK_MODEL_ID: identificador del modelo de lenguaje utilizado.

⚠️ Importante:
El archivo .env contiene credenciales sensibles y no debe subirse al repositorio.

4. Construcción y ejecución con Docker

Una vez configurado el archivo .env, se puede construir y ejecutar la aplicación mediante Docker Compose.

Desde el directorio raíz del proyecto:

docker compose up --build


Este comando:

Construye la imagen del backend

Crea los contenedores necesarios

Inicia automáticamente la aplicación

5. Acceso a la aplicación

Cuando los contenedores estén en ejecución, la aplicación estará disponible en:

Frontend (interfaz web):
👉 http://localhost:3000

Backend (API Flask):
👉 http://localhost:8000

Desde el navegador, el usuario podrá acceder al cuestionario y obtener recomendaciones personalizadas de ciclos formativos de FP.

6. Detener la aplicación

Para detener la ejecución de los contenedores:

docker compose down


O bien pulsar Ctrl + C en la terminal donde se esté ejecutando Docker Compose.

ℹ️ Información adicional

La aplicación integra Inteligencia Artificial generativa mediante Amazon Bedrock, cumpliendo con los requisitos del proyecto.

Las recomendaciones se basan en las respuestas del usuario y en datos reales de la oferta educativa de la Comunitat Valenciana.

El uso de Docker garantiza una ejecución sencilla y reproducible en distintos entornos.
