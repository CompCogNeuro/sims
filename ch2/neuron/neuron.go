// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
neuron: This simulation illustrates the basic properties of neural spiking and
rate-code activation, reflecting a balance of excitatory and inhibitory
influences (including leak and synaptic inhibition).
*/
package main

//go:generate core generate -add-types

import (
	"fmt"
	"log"
	"os"
	"reflect"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tree"
	"github.com/emer/emergent/v2/ecmd"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		sim.RunGUI()
	} else {
		sim.RunNoGUI()
	}
}

// see config.go for Config

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "no learning",
			Params: params.Params{
				"Path.Learn.Learn": "false",
			}},
		{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Inhib.Layer.On": "false",
				"Layer.Acts.Init.Vm":   "0.3",
			}},
	},
	"Testing": {
		{Sel: "Layer", Desc: "",
			Params: params.Params{
				"Layer.Acts.NMDA.Gbar":  "0.0",
				"Layer.Acts.GabaB.Gbar": "0.0",
			}},
	},
}

// Extra state for neuron
type NeuronEx struct {

	// input ISI countdown for spiking mode -- counts up
	InISI float32
}

func (nrn *NeuronEx) Init() {
	nrn.InISI = 0
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `display:"no-inline"`

	// extra neuron state for additional channels: VGCC, AK
	NeuronEx NeuronEx `display:"no-inline"`

	// leabra timing parameters and state
	Context leabra.Context

	// contains computed statistic values
	Stats estats.Stats

	// logging
	Logs elog.Logs `display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"inline"`

	// current cycle of updating
	Cycle int `edit:"-"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"inline"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// map of values for detailed debugging / testing
	ValMap map[string]float32 `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = leabra.NewNetwork("Neuron")
	econfig.Config(&ss.Config, "config.toml")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.ValMap = make(map[string]float32)
}

func (ss *Sim) Defaults() {
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	ctx := &ss.Context

	in := net.AddLayer2D("Input", 1, 1, leabra.InputLayer)
	hid := net.AddLayer2D("Neuron", 1, 1, leabra.SuperLayer)

	net.ConnectLayers(in, hid, paths.NewFull(), leabra.ForwardPath)

	err := net.Build(ctx)
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	ss.InitWeights(net)
}

// InitWeights loads the saved weights
func (ss *Sim) InitWeights(net *leabra.Network) {
	net.InitWeights(&ss.Context)
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Context.Reset()
	ss.InitWeights(ss.Net)
	ss.NeuronEx.Init()
	ss.GUI.StopNow = false
	ss.SetParams("", false) // all sheets
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Cycle:\t%d\t\t\t", ss.Context.Cycle)
}

func (ss *Sim) UpdateView() {
	ss.GUI.UpdatePlot(etime.Test, etime.Cycle)
	ss.GUI.ViewUpdate.Text = ss.Counters()
	ss.GUI.ViewUpdate.UpdateCycle(int(ss.Context.Cycle))
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// RunCycles updates neuron over specified number of cycles
func (ss *Sim) RunCycles() {
	ctx := &ss.Context
	ss.Init()
	ss.GUI.StopNow = false
	ss.Net.InitActs(ctx)
	ctx.NewState(etime.Train)
	ss.SetParams("", false)
	// ly := ss.Net.LayerByName("Neuron")
	// nrn := &(ly.Neurons[0])
	inputOn := false
	for cyc := 0; cyc < ss.Config.NCycles; cyc++ {
		switch cyc {
		case ss.Config.OnCycle:
			inputOn = true
		case ss.Config.OffCycle:
			inputOn = false
		}
		ss.NeuronUpdate(ss.Net, inputOn)
		ctx.Cycle = int32(cyc)
		ss.Logs.LogRow(etime.Test, etime.Cycle, cyc)
		ss.RecordValues(cyc)
		if cyc%ss.Config.UpdateInterval == 0 {
			ss.UpdateView()
		}
		ss.Context.CycleInc()
		if ss.GUI.StopNow {
			break
		}
	}
	ss.UpdateView()
}

func (ss *Sim) RecordValues(cyc int) {
	var vals []float32
	ly := ss.Net.LayerByName("Neuron")
	key := fmt.Sprintf("cyc: %03d", cyc)
	for _, vnm := range leabra.NeuronVarNames {
		ly.UnitValues(&vals, vnm, 0)
		vkey := key + fmt.Sprintf("\t%s", vnm)
		ss.ValMap[vkey] = vals[0]
	}
}

// NeuronUpdate updates the neuron
// this just calls the relevant code directly, bypassing most other stuff.
func (ss *Sim) NeuronUpdate(nt *leabra.Network, inputOn bool) {
	ctx := &ss.Context
	ly := ss.Net.LayerByName("Neuron")
	ni := ly.NeurStIndex
	di := uint32(0)
	ac := &ly.Params.Acts
	nex := &ss.NeuronEx
	// nrn.Noise = float32(ly.Params.Act.Noise.Gen(-1))
	// nrn.Ge += nrn.Noise // GeNoise
	// nrn.Gi = 0
	if inputOn {
		if ss.Config.GeClamp {
			leabra.SetNrnV(ctx, ni, di, leabra.GeRaw, ss.Config.Ge)
			leabra.SetNrnV(ctx, ni, di, leabra.GeSyn, ac.Dt.GeSynFromRawSteady(leabra.NrnV(ctx, ni, di, leabra.GeRaw)))
		} else {
			nex.InISI += 1
			if nex.InISI > 1000/ss.Config.SpikeHz {
				leabra.SetNrnV(ctx, ni, di, leabra.GeRaw, ss.Config.Ge)
				nex.InISI = 0
			} else {
				leabra.SetNrnV(ctx, ni, di, leabra.GeRaw, 0)
			}
			leabra.SetNrnV(ctx, ni, di, leabra.GeSyn, ac.Dt.GeSynFromRaw(leabra.NrnV(ctx, ni, di, leabra.GeSyn), leabra.NrnV(ctx, ni, di, leabra.GeRaw)))
		}
	} else {
		leabra.SetNrnV(ctx, ni, di, leabra.GeRaw, 0)
		leabra.SetNrnV(ctx, ni, di, leabra.GeSyn, 0)
	}
	leabra.SetNrnV(ctx, ni, di, leabra.GiRaw, ss.Config.Gi)
	leabra.SetNrnV(ctx, ni, di, leabra.GiSyn, ac.Dt.GiSynFromRawSteady(leabra.NrnV(ctx, ni, di, leabra.GiRaw)))

	if ss.Net.GPU.On {
		ss.Net.GPU.SyncStateToGPU()
		ss.Net.GPU.RunPipelineWait("Cycle", 2)
		ss.Net.GPU.SyncStateFromGPU()
		ctx.CycleInc() // why is this not working!?
	} else {
		lpl := ly.Pool(0, di)
		ly.GInteg(ctx, ni, di, lpl, ly.LayerValues(0))
		ly.SpikeFromG(ctx, ni, di, lpl)
	}
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.GUI.StopNow = true
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) {
	ss.Params.SetAll()
	ly := ss.Net.LayerByName("Neuron")
	lyp := ly.Params
	lyp.Acts.Gbar.E = 1
	lyp.Acts.Gbar.L = 0.2
	lyp.Acts.Erev.E = float32(ss.Config.ErevE)
	lyp.Acts.Erev.I = float32(ss.Config.ErevI)
	// lyp.Acts.Noise.Var = float64(ss.Config.Noise)
	lyp.Acts.KNa.On.SetBool(ss.Config.KNaAdapt)
	lyp.Acts.Mahp.Gbar = ss.Config.MahpGbar
	lyp.Acts.NMDA.Gbar = ss.Config.NMDAGbar
	lyp.Acts.GabaB.Gbar = ss.Config.GABABGbar
	lyp.Acts.VGCC.Gbar = ss.Config.VGCCGbar
	lyp.Acts.AK.Gbar = ss.Config.AKGbar
	lyp.Acts.Update()
}

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()

	ss.Logs.PlotItems("Vm", "Spike")

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
}

func (ss *Sim) ConfigLogItems() {
	ly := ss.Net.LayerByName("Neuron")
	// nex := &ss.NeuronEx
	lg := &ss.Logs

	lg.AddItem(&elog.Item{
		Name:   "Cycle",
		Type:   reflect.Int,
		FixMax: false,
		Range:  minmax.F32{Max: 1},
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
				ctx.SetInt(int(ss.Context.Cycle))
			}}})

	vars := []string{"GeSyn", "Ge", "Gi", "Inet", "Vm", "Act", "Spike", "Gk", "ISI", "ISIAvg", "VmDend", "GnmdaSyn", "Gnmda", "GABAB", "GgabaB", "Gvgcc", "VgccM", "VgccH", "Gak", "MahpN", "GknaMed", "GknaSlow", "GiSyn"}

	for _, vnm := range vars {
		cvnm := vnm // closure
		lg.AddItem(&elog.Item{
			Name:   cvnm,
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					vl := ly.UnitValue(cvnm, []int{0, 0}, 0)
					ctx.SetFloat32(vl)
				}}})
	}

}

func (ss *Sim) ResetTstCycPlot() {
	ss.Logs.ResetLog(etime.Test, etime.Cycle)
	ss.GUI.UpdatePlot(etime.Test, etime.Cycle)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	// nv.ViewDefaults()
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Neuron"
	ss.GUI.MakeBody(ss, "neuron", title, `This simulation illustrates the basic properties of neural spiking and rate-code activation, reflecting a balance of excitatory and inhibitory influences (including leak and synaptic inhibition). See <a href="https://github.com/emer/leabra/blob/master/examples/neuron/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.ConfigNetView(nv) // add labels etc
	ss.ViewUpdate.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)
	// key := etime.Scope(etime.Test, etime.Cycle)
	// plt := ss.GUI.NewPlot(key, ss.GUI.Tabs.NewTab("TstCycPlot"))
	// plt.SetTable(ss.Logs.Table(etime.Test, etime.Cycle))
	// egui.ConfigPlotFromLog("Neuron", plt, &ss.Logs, key)
	// ss.TstCycPlot = plt

	ss.GUI.Body.AddAppBar(func(p *tree.Plan) {
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Stop", Icon: icons.Stop,
			Tooltip: "Stops running.",
			Active:  egui.ActiveRunning,
			Func: func() {
				ss.Stop()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Run Cycles", Icon: icons.PlayArrow,
			Tooltip: "Runs neuron updating over NCycles.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					go func() {
						ss.GUI.IsRunning = true
						ss.RunCycles()
						ss.GUI.IsRunning = false
						ss.GUI.UpdateWindow()
					}()
				}
			},
		})
		tree.Add(p, func(w *core.Separator) {})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Plot", Icon: icons.Update,
			Tooltip: "Reset TstCycPlot.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.ResetTstCycPlot()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Defaults", Icon: icons.Update,
			Tooltip: "Restore initial default parameters.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Defaults()
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
			Icon:    "file-markdown",
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/leabra/blob/master/examples/neuron/README.md")
			},
		})
	})
	ss.GUI.FinalizeGUI(false)

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
		core.TheApp.AddQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}

func (ss *Sim) RunNoGUI() {
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name

	// netdata := ss.Config.Log.NetData
	// if netdata {
	// 	mpi.Printf("Saving NetView data from testing\n")
	// 	ss.GUI.InitNetData(ss.Net, 200)
	// }

	ss.Init()

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	ss.RunCycles()

	if ss.Config.Log.Cycle {
		dt := ss.Logs.Table(etime.Test, etime.Cycle)
		fnm := ecmd.LogFilename("cyc", netName, runName)
		dt.SaveCSV(core.Filename(fnm), table.Tab, table.Headers)
	}

	// if netdata {
	// 	ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	// }

	ss.Net.GPU.Destroy() // safe even if no GPU
}
