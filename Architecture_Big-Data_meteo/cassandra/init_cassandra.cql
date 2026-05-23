-- 1. Création de base de données
CREATE KEYSPACE IF NOT EXISTS weather_data 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 2};

-- On se place dans la base de données
USE weather_data;

-- 2. Table pour la BATCH LAYER (Statistiques Historiques)

CREATE TABLE IF NOT EXISTS city_stats (
    city text PRIMARY KEY,
    avg_temp_1h double;
    avg_temp_today double;
    max_temp_today double;
    last_update timestamp
);
USE weather_data;


-- 3. Table pour la SPEED LAYER (Alertes Temps Réel)
-- La clé primaire est composée du nom de la ville et du timestamp de l'alerte. Cela permet de stocker l'historique de toutes les alertes sans les écraser
CREATE TABLE IF NOT EXISTS weather_alerts (
    name text,
    alert_time timestamp,
    type text,
    PRIMARY KEY (name, alert_time, type)
) WITH CLUSTERING ORDER BY (alert_time DESC); 
