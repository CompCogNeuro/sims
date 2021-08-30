# Makefile for CCN sims projects

# important: update these for each app
ASSETS=faces.tsv partial_faces.tsv faces.wts

# default package install 
PKGDIR=$(HOME)/ccnsimpkg

# app is assumed to be the same as the directory name
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
penultimateword=$(wordlist $(words $1),$(words $1), x $1)

APP=$(current_dir)
CHAP=$(call penultimateword, $(subst /, ,$(dir $(mkfile_path))))
PLATFORM=$@
DEST=$(PKGDIR)/$(PLATFORM)/$(CHAP)

# all std go defs
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get

all: build

build: 
	$(GOBUILD) -v
dbg-build:
	$(GOBUILD) -v -gcflags=all="-N -l" -tags debug
test: 
	$(GOTEST) -v ./...
clean: 
	$(GOCLEAN)

# must do: go get github.com/shuLhan/go-bindata -- go install in cmd/go-bindata
bindata:
	go-bindata $(ASSETS)
mac: build
	- mkdir -p $(DEST)
	- /bin/cp $(APP) $(DEST)
linux: build
	- mkdir -p $(DEST)
	- /bin/cp $(APP) $(DEST)
windows: build
	- mkdir -p $(DEST)
	- cp $(APP).exe $(DEST)

