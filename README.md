
<h3>Crear entorno virtual</h3>

```
    >>> cd cnn_celular
    >>> python -m venv virtual
```

<h3>Activar entorno virtual</h3>

```
    >>> cd cnn_celular
    >>> virtual\Scripts\activate
```

<h3>Instalar dependencias</h3>

```
    >>> python.exe -m pip install --upgrade pip
    >>> pip install -r requirements.txt
```

<h3>Crear la base de datos</h3>

```
    >>> psql
    >>> CREATE USER cnn WITH PASSWORD 'root';
    >>> ALTER USER cnn CREATEDB;
    >>> ALTER USER cnn WITH SUPERUSER;
    >>> CREATE DATABASE cnn_model WITH OWNER = cnn;
    >>> GRANT ALL PRIVILEGES ON DATABASE cnn_model TO cnn;
    >>> COMMENT ON DATABASE cnn_model IS 'Base de datos de categorizacion de celulas';
```

<h3>Inicializar la base de datos</h3>

```
>>> cd /D ruta-proyecto
>>> flask db init
>>> flask db migrate -m "Initial migration"
>>> flask db upgrade
```

<h3>Actualiza base de datos</h3>

```
>>> cd /D ruta-proyecto
>>> flask db migrate -m "Comentario del cambio"
>>> flask db upgrade
```

<h3>Reiniciar datos (opcional)</h3>

```
# Borrar la carpeta /migrations
>>> cd /D ruta-proyecto
>>> flask shell
>>> from cnn.models.commons import db
>>> db.drop_all()
```