#!/usr/bin/env cosh

for _, pkg := range cosh.SplitLines(`go list ./...`) {
    pkg = strings.TrimPrefix(pkg, "github.com/CompCogNeuro/sims/v2")
    if pkg == "" {
        continue
    }
    pkg = pkg[1:] // remove slash
    core build web -dir {pkg} -o {filepath.Join("static", pkg)}
}
