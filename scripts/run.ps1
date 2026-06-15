param(
    [string]$Mode = "dev"
)

switch ($Mode) {

    "dev" {
        docker compose -f docker/dev/compose.yaml up --build
    }

    "prod" {
        docker compose -f docker/production/compose.yaml up --build -d
    }

    "stop" {
        docker compose -f docker/dev/compose.yaml down
        docker compose -f docker/production/compose.yaml down
    }

    default {
        Write-Host "Usage:"
        Write-Host "./scripts/run.ps1 dev"
        Write-Host "./scripts/run.ps1 prod"
        Write-Host "./scripts/run.ps1 stop"
    }
}