import json
import time
import urllib.request
from datetime import datetime
from kafka import KafkaProducer, KafkaClient 
from kafka.admin import KafkaAdminClient, NewTopic

def main():
    API_KEY = "558a3b9e35d409ae8c56ae81f3c01fa8"
    CITY = "Revelstoke"
    COUNTRY = "CA" 
    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY},{COUNTRY}&appid={API_KEY}&units=metric"

    # Configuration des brokers selon ton docker-compose (TP5)
    brokers = ["broker-1:9092", "broker-2:9092", "broker-3:9092"]
    topic = 'weather-data'
    
    # 1. Gestion de l'Admin Client pour le Topic
    try:
        admin = KafkaAdminClient(bootstrap_servers=brokers)
        server_topics = admin.list_topics()

        if topic not in server_topics:
            print("create new topic :", topic)
            # On utilise 3 réplications pour profiter des 3 brokers (Haute disponibilité)
            new_topic = NewTopic(name=topic,
                                 num_partitions=3, 
                                 replication_factor=3)
            admin.create_topics([new_topic])
        else:
            print(f"Topic {topic} est déjà créé")
    except Exception as e:
        print(f"Erreur avec l'AdminClient : {e}")

    # 2. Initialisation UNIQUE du Producer (Optimisée comme au TP5)
    producer = KafkaProducer(
        bootstrap_servers=brokers,
        # On sérialise en JSON automatiquement pour plus de propreté
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # 3. Boucle d'envoi du flux
    print(f"Début de l'envoi des données météo pour {CITY}...")
    try:
        while True:
            response = urllib.request.urlopen(url)
            data = json.loads(response.read().decode())
            
            # Envoi vers Kafka
            producer.send(topic, value=data)
            # On force l'envoi pour éviter que les messages restent dans le buffer
            producer.flush() 
            
            print("{} Produced weather record".format(datetime.now().strftime("%H:%M:%S")))
            
            # Pause de 180s (tu peux réduire à 10s pour tester ta Speed Layer plus vite)
            time.sleep(180)
            
    except KeyboardInterrupt:
        print("Arrêt du producer...")
    finally:
        producer.close()

if __name__ == "__main__":
    main()