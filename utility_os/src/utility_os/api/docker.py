from python_on_whales import docker

def get_images():
    return docker.images()

def inspect_image(id: str):
    return docker.image.inspect(id)

def get_containers():
    return docker.ps()

def inspect_container(id: str):
    return docker.containers.inspect(id)

def get_volumes():
    return docker.volume.list()

def get_system():
    docker_system = {}
    docker_system['disk'] = docker.system.disk_free()
    docker_system['info'] = docker.system.info()
    return docker_system
