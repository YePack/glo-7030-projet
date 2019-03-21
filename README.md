# glo-7030-projet

Dépot pour le projet de la session H2019 du cours GLO-7030.

# Création des étiquettes

Voici la procédure à suivre pour utiliser l'outil de *labeling* et créer les étiquettes pour les images.

## Prérequis

Docker doit être installé. Voir ce [lien](https://runnable.com/docker/install-docker-on-macos) pour installation sur macOS. 

##  Accéder à l'outil

1. Mettre en marche Docker
2. Aller dans le répertoire `lableling-tool/`
3. *Builder* l'image docker : `docker-compose build`
4. Mettre en marche le *container* : `docker-compose up -d`
5. Créer un superuser pour avoir accès à tous les droits : `docker exec -it cvat bash -ic '/usr/bin/python3 ~/manage.py createsuperuser'`
6. Vous devriez maintenant pouvoir accéder à l'outil via [ce lien](http://localhost:8080/auth/register).
