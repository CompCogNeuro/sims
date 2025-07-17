// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/CompCogNeuro/sims/v2/ch4/associator"
	"github.com/emer/emergent/v2/egui"
)

func main() { egui.Run[associator.Sim, associator.Config]() }
