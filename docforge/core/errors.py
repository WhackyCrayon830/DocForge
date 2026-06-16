class DocForgeError(Exception):
    pass

class ConfigError(DocForgeError):
    pass

class ProjectError(DocForgeError):
    pass

class IngestError(DocForgeError):
    pass

class StoreError(DocForgeError):
    pass