// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// The code in this file was listed verbatim from the cemer version of PVLV, and tries to duplicate its logic fathfully.

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"cogentcore.org/core/kit"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/leabra/v2/examples/pvlv/data"
	"github.com/emer/leabra/v2/pvlv"
)

type PVLVEnv struct {
	Nm                 string                  `inactive:"+" desc:"name of this environment"`
	Dsc                string                  `inactive:"+" desc:"description of this environment"`
	PVLVParams         *params.Params          `desc:"PVLV-specific params"`
	GlobalStep         int                     `desc:"cycle counter, cleared by Init, otherwise increments on every Cycle"`
	MultiRunCt         env.Ctr                 `inactive:"+" view:"inline" desc:"top-level counter for multi-run sequence"`
	ConditionCt        env.Ctr                 `inactive:"+" view:"inline" desc:"top-level counter for multi-trial group run"`
	TrialBlockCt       env.Ctr                 `inactive:"+" view:"inline" desc:"trial group within a block"`
	TrialCt            env.Ctr                 `inactive:"+" view:"inline" desc:"trial group within a set of trial groups"`
	AlphaCycle         env.Ctr                 `inactive:"+" view:"inline" desc:"step within a trial"`
	AlphaTrialName     string                  `inactive:"+" desc:"name of current alpha trial step"`
	USTimeInStr        string                  `inactive:"+" desc:"decoded value of USTimeIn"`
	TrialBlockParams   *data.TrialBlockRecs    `desc:"AKA trial group list. A set of trial groups to be run together"`
	TrialInstances     *data.TrialInstanceRecs //*TrialInstanceList `view:"no-inline" desc:"instantiated trial groups, further unpacked into StdInputData from this"`
	StdInputData       *etable.Table           `desc:"Completely instantiated input data for a single block"`
	ContextModel       ContextModel            `inactive:"+" desc:"One at a time, conjunctive, or a mix"`
	SeqRun             bool                    `view:"-" desc:"running from a top-level sequence?"`
	CurConditionParams *data.ConditionParams   `view:"-" desc:"params for currently executing block, whether from selection or sequence"`
	TrialsPerBlock     int                     `inactive:"+"`
	DataLoopOrder      data.DataLoopOrder      `inactive:"+"`
	BlockEnded         bool                    `view:"-"`

	// Input data tensors
	TsrStimIn    etensor.Float64
	TsrPosPV     etensor.Float64
	TsrNegPV     etensor.Float64
	TsrContextIn etensor.Float64
	TsrUSTimeIn  etensor.Float64

	NormContextTotalAct   bool    `view:"-" `                                                       // TODO UNUSED if true, clamp ContextIn units as 1/n_context_units - reflecting mutual competition
	NormStimTotalAct      bool    `view:"-" `                                                       // TODO UNUSED if true, clamp StimIn units as 1/n_context_units - reflecting mutual competition
	NormUSTimeTotalAct    bool    `view:"-" `                                                       // TODO UNUSED if true, clamp USTimeIn units as 1/n_context_units - reflecting mutual competition
	PctNormTotalActStim   float64 `desc:"amount to add to denominator for StimIn normalization"`    // used in InstantiateBlockTrials and SetRowStdInputDataAlphTrial
	PctNormTotalActCtx    float64 `desc:"amount to add to denominator for ContextIn normalization"` // used in InstantiateBlockTrials and SetRowStdInputDataAlphTrial
	PctNormTotalActUSTime float64 `desc:"amount to add to denominator for USTimeIn normalization"`  // used in InstantiateBlockTrials and SetRowStdInputDataAlphTrial

	InputShapes *map[string][]int
}

func (ev *PVLVEnv) Name() string { return ev.Nm }
func (ev *PVLVEnv) Desc() string { return ev.Dsc }

func (ev *PVLVEnv) New(ss *Sim) {
	ev.CurConditionParams = ss.ConditionParams
	ev.ConditionCt.Scale = env.Epoch // We don't really have the right term available
	ev.TrialBlockCt.Scale = env.Block
	ev.AlphaCycle.Scale = env.Trial
	ev.InputShapes = &ss.InputShapes
	ev.ContextModel = ss.ContextModel // lives in MiscParams in cemer
	ev.AlphaCycle.Init()
	ev.StdInputData = &etable.Table{}
	ev.ConfigStdInputData(ev.StdInputData)
	ev.AlphaTrialName = "trialname"
	ev.TsrStimIn.SetShape(ss.InputShapes["StimIn"], nil, nil)
	ev.TsrContextIn.SetShape(ss.InputShapes["ContextIn"], nil, nil)
	ev.TsrUSTimeIn.SetShape(ss.InputShapes["USTimeIn"], nil, nil)
	ev.TsrPosPV.SetShape(ss.InputShapes["PosPV"], nil, nil)
	ev.TsrNegPV.SetShape(ss.InputShapes["NegPV"], nil, nil)
}

// From looking at the running cemer model, the chain is as follows:
// RunParams(=pos_acq) -> ConditionParams(=pos_acq_b50) -> PVLVEnv.vars["env_params_table"](=PosAcq_B50)
// RunParams fields: seq_step_1...5 from Run.vars
// ConditionParams fields: env_params_table, fixed_prob, ... lrs_bump_step, n_batches, batch_start, load_exp, pain_exp
// Trial fields: trial_gp_name, percent_of_total, ...
func (ev *PVLVEnv) Init(ss *Sim, firstCondition bool) (ok bool) {
	ev.CurConditionParams = ss.ConditionParams
	ev.TrialBlockParams, ok = ss.GetTrialBlockParams(ev.CurConditionParams.TrialBlkNm)
	if !ok {
		fmt.Printf("TrialBlockParams lookup failed for %v\n", ev.CurConditionParams.TrialBlkNm)
		return ok
	}
	if firstCondition {
		ev.ConditionCt.Init()
		ev.ConditionCt.Max = ss.MaxConditions
	}
	ev.TrialBlockCt.Init()
	ev.TrialBlockCt.Max = ev.CurConditionParams.NIters
	ev.TrialInstances = data.NewTrialInstanceRecs(nil)
	ev.TrialCt.Init()
	ev.TrialCt.Max = ev.CurConditionParams.BlocksPerIter
	ev.AlphaCycle.Init()
	ev.ContextModel = ss.ContextModel // lives in MiscParams in cemer
	return ok
}

func (ev *PVLVEnv) ConfigStdInputData(dt *etable.Table) {
	dt.SetMetaData("name", "StdInputData")
	dt.SetMetaData("desc", "input data")
	dt.SetMetaData("precision", "6")
	shapes := *ev.InputShapes
	sch := etable.Schema{
		{"AlphTrialName", etensor.STRING, nil, nil},
		{"Stimulus", etensor.STRING, nil, nil},
		{"Time", etensor.STRING, nil, nil},
		{"Context", etensor.STRING, nil, nil},
		{"USTimeInStr", etensor.STRING, nil, nil},
		{"PosPV", etensor.FLOAT64, shapes["PosPV"], nil},
		{"NegPV", etensor.FLOAT64, shapes["NegPV"], nil},
		{"StimIn", etensor.FLOAT64, shapes["StimIn"], nil},
		{"ContextIn", etensor.FLOAT64, shapes["ContextIn"], nil},
		{"USTimeIn", etensor.FLOAT64, shapes["USTimeIn"], nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ev *PVLVEnv) Defaults() {
	// how much to decrease clamped input activity values when multiple stimuli present (due to presumed mutual competition), e.g., for AX trials: 0 = not at all; 1 = full activity normalization
	//ev.NormProb = 0.5

}

// BlockStart
func (ev *PVLVEnv) BlockStart(ss *Sim) {
	for colNm := range ss.TrialTypeBlockFirstLogged {
		ss.TrialTypeBlockFirstLogged[colNm] = false
	}
	ev.AlphaCycle.Init()
	ev.TrialInstances = data.NewTrialInstanceRecs(nil)
	ev.TrialCt.Init()
	ss.Net.ThrTimerReset()
}

// end BlockStart (no internal functions)

// BlockEnd
func (ev *PVLVEnv) BlockEnd(ss *Sim) {
	//ss.TrialAnalysis(ev)
	ss.BlockMonitor()
	if ev.TrialBlockCt.Cur%ev.CurConditionParams.SaveWtsInterval == 0 && ev.TrialBlockCt.Cur > 0 {
		ev.SaveWeights(ss)
	}
}

// end BlockEnd

// SaveWeights
func (ev *PVLVEnv) SaveWeights(_ *Sim) {
	// TODO implement SaveWeights
}

// end SaveWeights

// ContextModel
type ContextModel int

const (
	ELEMENTAL ContextModel = iota
	CONJUNCTIVE
	BOTH
	ContextModelN
)

var KiT_ContextModel = kit.Enums.AddEnum(ContextModelN, kit.NotBitFlag, nil)

func (ev *PVLVEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Epoch, env.Block, env.Trial}
}

func (ev *PVLVEnv) Actions() env.Elements {
	return nil
}

func (ev *PVLVEnv) States() env.Elements {
	// one-hot representations for each component
	shapes := *ev.InputShapes
	els := env.Elements{
		{"StimIn", shapes["StimIn"], nil}, //[]string{"N"}},
		{"ContextIn", shapes["ContextIn"], []string{"Ctx", "Time"}},
		{"USTimeIn", shapes["USTimeIn"],
			[]string{"CS", "Valence", "Time", "US"}},
		{"PosPV", shapes["PosPV"], []string{"PV"}},
		{"NegPV", shapes["NegPV"], []string{"PV"}},
	}
	return els
}

// SetState sets the input states from ev.StdInputData
func (ev *PVLVEnv) SetState() {
	ev.TsrStimIn.SetZeros()
	ev.TsrStimIn.CopyFrom(ev.StdInputData.CellTensor("StimIn", ev.AlphaCycle.Cur))
	ev.TsrContextIn.SetZeros()
	ev.TsrContextIn.CopyFrom(ev.StdInputData.CellTensor("ContextIn", ev.AlphaCycle.Cur))
	ev.TsrUSTimeIn.SetZeros()
	ev.TsrUSTimeIn.CopyFrom(ev.StdInputData.CellTensor("USTimeIn", ev.AlphaCycle.Cur))
	ev.TsrPosPV.SetZeros()
	ev.TsrPosPV.CopyFrom(ev.StdInputData.CellTensor("PosPV", ev.AlphaCycle.Cur))
	ev.TsrNegPV.SetZeros()
	ev.TsrNegPV.CopyFrom(ev.StdInputData.CellTensor("NegPV", ev.AlphaCycle.Cur))
	ev.AlphaTrialName = ev.StdInputData.CellString("AlphTrialName", ev.AlphaCycle.Cur)
	ev.USTimeInStr = ev.StdInputData.CellString("USTimeInStr", ev.AlphaCycle.Cur)
}

func (ev *PVLVEnv) State(Nm string) etensor.Tensor {
	switch Nm {
	case "StimIn":
		return &ev.TsrStimIn
	case "ContextIn":
		return &ev.TsrContextIn
	case "USTimeIn":
		return &ev.TsrUSTimeIn
	case "PosPV":
		return &ev.TsrPosPV
	case "NegPV":
		return &ev.TsrNegPV
	default:
		return nil
	}
}

//func (ev *PVLVEnv) Step() bool {
//	ev.TrialBlockCt.Same() // good idea to just reset all non-inner-most counters at start
//	if ev.AlphaCycle.Incr() {
//		ev.TrialBlockCt.Incr()
//	}
//	return true
//}

func (ev *PVLVEnv) Action(_ string, _ etensor.Tensor) {
	// nop
}

func (ev *PVLVEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Block:
		return ev.ConditionCt.Query()
	case env.Epoch:
		return ev.TrialBlockCt.Query()
	case env.Trial:
		return ev.AlphaCycle.Query()
	}
	return -1, -1, false
}

// normalizes PercentOfTotal numbers for trial types within a set
// sets X labels in the TrialTypeData plot from instantiated trial types
// ev.SetTableTrialGpListFmDefnTable to set up a full blocks's worth of trials
// permutes trial order, if specified
func (ev *PVLVEnv) SetActiveTrialList(ss *Sim) {
	ev.TrialInstances.Reset()
	pctTotalSum := 0.0

	ev.TrialBlockParams.Reset()
	for !ev.TrialBlockParams.AtEnd() {
		pctTotalSum += ev.TrialBlockParams.ReadNext().PercentOfTotal
	}

	ev.TrialBlockParams.Reset()
	for !ev.TrialBlockParams.AtEnd() {
		tg := ev.TrialBlockParams.ReadNext()
		baseNumber := tg.PercentOfTotal
		normProb := baseNumber / pctTotalSum
		tg.PercentOfTotal = normProb
	}

	ev.SetTableTrialGpListFmDefnTable()

	if ev.CurConditionParams.PermuteTrialGps {
		ev.TrialInstances.Permute()
		ev.TrialInstances.Reset()
	}
}

// turn US probabilities into concrete examples
// reads from ev.TrialBlock, writes to ev.TrialInstances
func (ev *PVLVEnv) SetTableTrialGpListFmDefnTable() {
	nRepeatsF := 0.0
	nRepeats := 0
	usFlag := false
	exactOmitProportion := false
	exactNOmits := 0
	nOmitCount := 0

	ev.TrialBlockParams.SetOrder(ev.DataLoopOrder)
	for !ev.TrialBlockParams.AtEnd() {
		curBlockParams := ev.TrialBlockParams.ReadNext()
		//var curTrial int

		nRepeatsF = curBlockParams.PercentOfTotal * float64(ev.CurConditionParams.BlocksPerIter)
		// fix rounding error from int arithmetic
		nRepeats = int(nRepeatsF + 0.001)
		if nRepeats < 1 && curBlockParams.PercentOfTotal > 0.0 {
			nRepeats = 1
		}
		// should do each at least once (unless user intended 0.0f)
		if ev.CurConditionParams.FixedProb || strings.Contains(curBlockParams.TrialBlkName, "AutoTstEnv") {
			if curBlockParams.USProb != 0.0 && curBlockParams.USProb != 1.0 {
				exactOmitProportion = true
				exactNOmits = int(math.Round(float64(ev.CurConditionParams.BlocksPerIter) * curBlockParams.PercentOfTotal * (1.0 - curBlockParams.USProb)))
				nOmitCount = 0
			} else {
				exactOmitProportion = false
			}
		}
		if ev.TrialBlockParams.AtEnd() && ev.TrialInstances.Cur()+nRepeats < ev.TrialCt.Max {
			nRepeats++
		}
		for i := 0; i < nRepeats; i++ {
			if ev.TrialInstances != nil &&
				(ev.TrialInstances.Length() < ev.CurConditionParams.BlocksPerIter) {
				// was SetRow_CurEpoch
				rfFlgTemp := false
				probUSOmit := 0.0
				trialGpName := curBlockParams.TrialBlkName + "_" + curBlockParams.ValenceContext.String()
				if !strings.Contains(trialGpName, "NR") { // nonreinforced (NR) trials NEVER get reinforcement
					rfFlgTemp = true // actual Rf can be different each eco_trial
					if !exactOmitProportion {
						probUSOmit = rand.Float64()
						if probUSOmit >= curBlockParams.USProb {
							rfFlgTemp = false
						}
					} else {
						if nOmitCount < exactNOmits {
							rfFlgTemp = false
							nOmitCount++
						} else {
							rfFlgTemp = true
						}
					}
					trialGpName = strings.TrimSuffix(trialGpName, "_omit")
					// could be repeat of eco trial type - but with different Rf flag
					if rfFlgTemp == false {
						trialGpName += "_omit"
					}
					usFlag = rfFlgTemp
				} else {
					usFlag = false
				}
				trialGpName = strings.TrimSuffix(trialGpName, "_test")
				// could be repeat of eco trial type - but with different test flag
				testFlag := false
				if strings.Contains(trialGpName, "_test") {
					parts := strings.Split(trialGpName, "_")
					trialGpName = ""
					for _, part := range parts {
						if part != "test" {
							trialGpName += part + "_"
						}
					}
					trialGpName += "test"
					testFlag = true // just in case - should be redundant
				} else {
					testFlag = false
				}
				curTrial := new(data.TrialInstance)
				curTrial.TrialName = trialGpName
				curTrial.ValenceContext = curBlockParams.ValenceContext
				curTrial.TestFlag = testFlag
				curTrial.USFlag = usFlag
				curTrial.MixedUS = curBlockParams.MixedUS
				curTrial.USProb = curBlockParams.USProb
				curTrial.USMagnitude = curBlockParams.USMagnitude
				if curTrial.ValenceContext == pvlv.POS {
					curTrial.USType = pvlv.PosUS(curBlockParams.USType).String()
				} else {
					curTrial.USType = pvlv.NegUS(curBlockParams.USType).String()
				}
				curTrial.AlphaTicksPerTrialGp = curBlockParams.AlphTicksPerTrialGp
				curTrial.CS = curBlockParams.CS
				curTrial.CSTimeStart = int(curBlockParams.CSTimeStart)
				curTrial.CSTimeEnd = int(curBlockParams.CSTimeEnd)
				curTrial.CS2TimeStart = int(curBlockParams.CS2TimeStart)
				curTrial.CS2TimeEnd = int(curBlockParams.CS2TimeEnd)
				curTrial.USTimeStart = int(curBlockParams.USTimeStart)
				curTrial.USTimeEnd = int(curBlockParams.USTimeEnd)
				curTrial.Context = curBlockParams.Context
				ev.TrialInstances.WriteNext(curTrial)
				// end SetRow_CurEpoch
			}
		}
	}
	ev.TrialBlockParams.Sequential() // avoid confusion?
}

func (ev *PVLVEnv) SetupOneAlphaTrial(curTrial *data.TrialInstance, stimNum int) {
	prefixUSTimeIn := ""

	// CAUTION! - using percent normalization assumes the multiple CSs (e.g., AX) are always on together,
	// i.e., the same timesteps; thus, doesn't work for second-order conditioning
	stimInBase := pvlv.StmNone
	stimIn2Base := pvlv.StmNone
	nStims := 1
	nUSTimes := 1

	// CAUTION! below string-pruning requires particular convention for naming trial_gps in terms of CSs used;
	// e.g., "AX_*", etc.

	// CAUTION! For either multiple CSs (e.g., AX) or mixed_US case
	// (e.g., sometimes reward, sometimes punishment) only two (2) simultaneous representations
	// currently supported; AND, multiple CSs and mixed_US cases can only be used separately, not together;
	// code will need re-write if more complicated cases are desired (e.g., more than two (2) representations
	// or using multiple CSs/mixed_US together).

	cs := curTrial.TrialName[0:2]
	cs1 := ""
	cs2 := ""
	if strings.Contains(cs, "_") {
		cs1 = cs[0:1]
		cs = pvlv.StmNone.String()
		cs2 = pvlv.StmNone.String()
		nStims = 1
		// need one for each predictive CS; also, one for each PREDICTED US if same CS (e.g., Z')
		// predicts two different USs probalistically (i.e., mixed_US == true condition)
		nUSTimes = 1
		stimInBase = pvlv.StimMap[cs1]
	} else {
		cs1 = cs[0:1]
		cs2 = cs[1:2]
		nStims = 2
		// need one for each predictive CS; also, one for each PREDICTED US if same CS (e.g., Z')
		// predicts two different USs probalistically (i.e., mixed_US == true condition)
		nUSTimes = 2
		stimInBase = pvlv.StimMap[cs1]
		stimIn2Base = pvlv.StimMap[cs2]
	}

	// Set up Context_In reps

	// initialize to use the basic context_in var to rep the basic case in which CS and Context are isomorphic
	ctxParts := pvlv.CtxRe.FindStringSubmatch(curTrial.Context)
	ctx1 := ctxParts[1]
	ctx2 := ctxParts[2]
	preContext := ctx1 + ctx2
	postContext := ctxParts[3]
	contextIn := pvlv.CtxMap[curTrial.Context]
	contextIn2 := pvlv.CtxNone
	contextIn3 := pvlv.CtxNone
	nContexts := len(preContext)
	// gets complicated if more than one CS...
	if len(preContext) > 1 {
		switch ev.ContextModel {
		case ELEMENTAL:
			// first element, e.g., A
			contextIn = pvlv.CtxMap[ctx1]
			// second element, e.g., X
			contextIn2 = pvlv.CtxMap[ctx2]
			// only handles two for now...
		case CONJUNCTIVE:
			// use "as is"...
			contextIn = pvlv.CtxMap[curTrial.Context]
			nContexts = 1
		case BOTH:
			// first element, e.g., A
			contextIn = pvlv.CtxMap[ctx1]
			// second element, e.g., X
			contextIn2 = pvlv.CtxMap[ctx2]
			// conjunctive case, e.g., AX
			contextIn3 = pvlv.CtxMap[preContext]
			nContexts = len(preContext) + 1
		}
	}
	// anything after the "_" indicates different context for extinction, renewal, etc.
	if len(postContext) > 0 {
		contextIn = pvlv.CtxMap[ctx1+"_"+postContext]
		if len(ctx2) > 0 {
			contextIn2 = pvlv.CtxMap[ctx2+"_"+postContext]
		}
		contextIn3 = pvlv.CtxNone
	}

	if ev.StdInputData.Rows != 0 {
		ev.StdInputData.SetNumRows(0)
	}

	// configure and write all the leabra trials for one eco trial
	for i := 0; i < curTrial.AlphaTicksPerTrialGp; i++ {
		i := ev.AlphaCycle.Cur
		alphaTrialName := curTrial.TrialName + "_t" + strconv.Itoa(i)
		trialGpTimestep := pvlv.Tick(i)
		trialGpTimestepInt := i
		stimIn := pvlv.StmNone
		stimIn2 := pvlv.StmNone
		posPV := pvlv.PosUSNone
		negPV := pvlv.NegUSNone
		usTimeInStr := ""
		usTimeIn2Str := ""
		usTimeInWrongStr := ""
		usTimeIn := pvlv.USTimeNone
		usTimeIn2 := pvlv.USTimeNone
		usTimeInWrong := pvlv.USTimeNone
		notUSTimeIn := pvlv.USTimeNone
		prefixUSTimeIn = cs1 + "_"
		prefixUSTimeIn2 := ""
		if nUSTimes == 2 {
			prefixUSTimeIn2 = cs2 + "_"
		}
		// set CS input activation values on or off according to timesteps
		// set first CS - may be the only one
		if i >= curTrial.CSTimeStart && i <= curTrial.CSTimeEnd {
			stimIn = stimInBase
			// TODO: Theoretically, USTime reps shouldn't come on at CS-onset until BAacq and/or
			// gets active first - for time being, using a priori inputs as a temporary proof-of-concept
		} else {
			stimIn = pvlv.StmNone
		}
		// set CS2 input activation values on or off according to timesteps, if a second CS exists
		if i >= curTrial.CS2TimeStart && i <= curTrial.CS2TimeEnd {
			stimIn2 = stimIn2Base
		} else {
			stimIn2 = pvlv.StmNone
		}
		// set US and USTime input activation values on or off according to timesteps
		var us int
		if i > curTrial.CSTimeStart && (!(i > curTrial.USTimeStart) || !curTrial.USFlag) {
			if curTrial.ValenceContext == pvlv.POS {
				us = int(pvlv.PosSMap[curTrial.USType])
				posPV = pvlv.PosUS(us)
				usTimeInStr = prefixUSTimeIn + "PosUS" + strconv.Itoa(us) + "_t" +
					strconv.Itoa(i-curTrial.CSTimeStart-1)
				usTimeIn = pvlv.PUSTFromString(usTimeInStr)
				usTimeInWrongStr = pvlv.USTimeNone.String()
				if curTrial.MixedUS {
					usTimeInWrongStr = prefixUSTimeIn + "NegUS" + strconv.Itoa(us) + "_t" +
						strconv.Itoa(i-curTrial.CSTimeStart-1)
					usTimeInWrong = pvlv.PUSTFromString(usTimeInWrongStr)
				}
			} else if curTrial.ValenceContext == pvlv.NEG {
				us = int(pvlv.NegSMap[curTrial.USType])
				negPV = pvlv.NegUS(us)
				usTimeInStr = prefixUSTimeIn + "NegUS" + strconv.Itoa(us) + "_t" +
					strconv.Itoa(i-curTrial.CSTimeStart-1)
				usTimeIn = pvlv.PUSTFromString(usTimeInStr)
				usTimeInWrongStr = pvlv.USTimeNone.String()
				if curTrial.MixedUS {
					usTimeInWrongStr = prefixUSTimeIn + "PosUS" + strconv.Itoa(us) + "_t" +
						strconv.Itoa(i-curTrial.CSTimeStart-1)
					usTimeInWrong = pvlv.PUSTFromString(usTimeInWrongStr)
				}
			}
		} else {
			usTimeIn = pvlv.USTimeNone
			notUSTimeIn = pvlv.USTimeNone
			usTimeInStr = pvlv.USTimeNone.String()
		}

		if i > curTrial.CS2TimeStart && i <= (curTrial.CS2TimeEnd+1) && (!(i > curTrial.USTimeStart) || !curTrial.USFlag) {
			usTime2IntStr := strconv.Itoa(i - curTrial.CS2TimeStart - 1)
			if curTrial.ValenceContext == pvlv.POS {
				us = int(pvlv.PosSMap[curTrial.USType])
				posPV = pvlv.PosUS(us)
				usTimeIn2Str = prefixUSTimeIn2 + "PosUS" + strconv.Itoa(us) + "_t" + usTime2IntStr
				usTimeIn2 = pvlv.PUSTFromString(usTimeIn2Str)
				usTimeInWrongStr = pvlv.USTimeNone.String()
				if curTrial.MixedUS {
					usTimeInWrongStr = prefixUSTimeIn + "NegUS" + strconv.Itoa(us) + "_t" + usTime2IntStr
					usTimeInWrong = pvlv.USTimeNone.FromString(usTimeInWrongStr)
				}
			} else if curTrial.ValenceContext == pvlv.NEG {
				negPV = pvlv.NegSMap[curTrial.USType]
				us = int(negPV)
				usTimeIn2Str = prefixUSTimeIn2 + "NegUS" + strconv.Itoa(us) + "_t" + usTime2IntStr
				usTimeIn2 = pvlv.PUSTFromString(usTime2IntStr)
				usTimeInWrongStr = pvlv.USTimeNone.String()
				if curTrial.MixedUS {
					usTimeInWrongStr = prefixUSTimeIn + "PosUS" + strconv.Itoa(us) + "_t" +
						strconv.Itoa(i-curTrial.CSTimeStart-1)
					usTimeInWrong = pvlv.USTimeNone.FromString(usTimeInWrongStr)
				}
			}
		} else {
			usTimeIn2 = pvlv.USTimeNone
			notUSTimeIn = pvlv.USTimeNone
			usTimeIn2Str = pvlv.USTimeNone.String()
		}

		if (i >= curTrial.USTimeStart) && (i <= curTrial.USTimeEnd) && curTrial.USFlag {
		} else {
			posPV = pvlv.PosUSNone
			negPV = pvlv.NegUSNone
		}
		if (i > curTrial.USTimeStart) && curTrial.USFlag {
			if curTrial.ValenceContext == pvlv.POS {
				us = int(pvlv.PosSMap[curTrial.USType])
				usTimeInStr = "PosUS" + strconv.Itoa(us) + "_t" + strconv.Itoa(i-curTrial.USTimeStart-1)
				usTimeIn = pvlv.USTimeNone.FromString(usTimeInStr)
				usTimeInWrongStr = pvlv.USTimeNone.String()
				usTimeInWrong = pvlv.USTimeNone
			} else if curTrial.ValenceContext == pvlv.NEG {
				us = int(pvlv.NegSMap[curTrial.USType])
				usTimeInStr = "NegUS" + strconv.Itoa(us) + "_t" + strconv.Itoa(i-curTrial.USTimeStart-1)
				usTimeIn = pvlv.USTimeNone.FromString(usTimeInStr)
				usTimeInWrongStr = pvlv.USTimeNone.String()
				usTimeInWrong = pvlv.USTimeNone
			}
		}
		pvEmpty := pvlv.PosUSNone.Tensor()
		curTimestepStr := ""
		curTimeStepInt := 0
		stimulus := ""
		stimDenom := 1.0
		ctxtDenom := 1.0
		usTimeDenom := 1.0
		curTimeStepInt = trialGpTimestepInt
		curTimestepStr = trialGpTimestep.String()
		if stimNum == 0 {
			ev.StdInputData.AddRows(1)
		}
		if nStims == 1 {
			stimulus = stimIn.String()
		} else {
			stimulus = cs1 + cs2
		} // // i.e., there is a 2nd stimulus, e.g., 'AX', 'BY'

		ev.StdInputData.SetCellString("AlphTrialName", curTimeStepInt, alphaTrialName)
		ev.StdInputData.SetCellString("Time", curTimeStepInt, curTimestepStr)
		ev.StdInputData.SetCellString("Stimulus", curTimeStepInt, stimulus)
		ev.StdInputData.SetCellString("Context", curTimeStepInt, curTrial.Context)

		tsrStim := etensor.NewFloat64(pvlv.StimInShape, nil, nil)
		tsrCtx := etensor.NewFloat64(pvlv.ContextInShape, nil, nil)
		if curTimeStepInt >= curTrial.CSTimeStart && curTimeStepInt <= curTrial.CSTimeEnd {
			stimDenom = 1.0 + ev.PctNormTotalActStim*float64(nStims-1)
			if stimIn != pvlv.StmNone {
				tsrStim.SetFloat([]int{int(stimIn)}, 1.0/stimDenom)
			}
			if stimIn2 != pvlv.StmNone {
				tsrStim.SetFloat([]int{int(stimIn2)}, 1.0/stimDenom)
			}
			ev.StdInputData.SetCellTensor("StimIn", curTimeStepInt, tsrStim)

			ctxtDenom = 1.0 + ev.PctNormTotalActCtx*float64(nContexts-1)
			if contextIn != pvlv.CtxNone {
				tsrCtx.SetFloat(contextIn.Parts(), 1.0/ctxtDenom)
			}
			if contextIn3 != pvlv.CtxNone {
				tsrCtx.SetFloat(contextIn3.Parts(), 1.0/ctxtDenom)
			}
			ev.StdInputData.SetCellTensor("ContextIn", curTimeStepInt, tsrCtx)
		}
		if curTimeStepInt >= curTrial.CS2TimeStart && curTimeStepInt <= curTrial.CS2TimeEnd {
			stimDenom = 1.0 + ev.PctNormTotalActStim*float64(nStims-1)
			if stimIn2 != pvlv.StmNone {
				tsrStim.SetFloat([]int{int(stimIn2)}, 1.0/stimDenom)
			}
			ev.StdInputData.SetCellTensor("StimIn", curTimeStepInt, tsrStim)

			ctxtDenom = 1.0 + ev.PctNormTotalActCtx*float64(nContexts-1)
			if contextIn2 != pvlv.CtxNone {
				tsrCtx.SetFloat(contextIn2.Parts(), 1.0/ctxtDenom)
			}
			if contextIn3 != pvlv.CtxNone {
				tsrCtx.SetFloat(contextIn3.Parts(), 1.0/ctxtDenom)
			}
			ev.StdInputData.SetCellTensor("ContextIn", curTimeStepInt, tsrCtx)
		}

		if curTrial.USFlag && (curTimeStepInt >= curTrial.USTimeStart && curTimeStepInt <= curTrial.USTimeEnd) {
			if curTrial.USFlag && curTrial.ValenceContext == pvlv.POS {
				if posPV != pvlv.PosUSNone {
					ev.StdInputData.SetCellTensor("PosPV", curTimeStepInt, posPV.Tensor())
				} else {
					ev.StdInputData.SetCellTensor("PosPV", curTimeStepInt, pvEmpty)
				}
			} else if curTrial.USFlag && curTrial.ValenceContext == pvlv.NEG {
				if negPV != pvlv.NegUSNone {
					ev.StdInputData.SetCellTensor("NegPV", curTimeStepInt, negPV.Tensor())
				} else {
					ev.StdInputData.SetCellTensor("NegPV", curTimeStepInt, pvEmpty)
				}
			}
		} else {
			ev.StdInputData.SetCellTensor("PosPV", curTimeStepInt, pvEmpty)
			ev.StdInputData.SetCellTensor("NegPV", curTimeStepInt, pvEmpty)
		}

		usTimeDenom = 1.0 + ev.PctNormTotalActUSTime*float64(nUSTimes-1)
		tsrUSTime := etensor.NewFloat64(pvlv.USTimeInShape, nil, nil)
		if usTimeIn != pvlv.USTimeNone {
			setVal := usTimeIn.Unpack().Coords()
			tsrUSTime.SetFloat(setVal, 1.0/usTimeDenom)
		}
		if usTimeIn2 != pvlv.USTimeNone {
			setVal := usTimeIn2.Unpack().Coords()
			tsrUSTime.SetFloat(setVal, 1.0/usTimeDenom)
		}
		if usTimeInWrong != pvlv.USTimeNone {
			tsrUSTime.SetFloat(usTimeInWrong.Shape(), 1.0/usTimeDenom)
		}
		if notUSTimeIn != pvlv.USTimeNone {
			tsrUSTime.SetFloat(notUSTimeIn.Shape(), 1.0/usTimeDenom)
		}
		ev.StdInputData.SetCellTensor("USTimeIn", curTimeStepInt, tsrUSTime)
		if usTimeIn2Str != "" {
			usTimeIn2Str = "+" + usTimeIn2Str + usTimeIn2.Unpack().CoordsString()
		}
		ev.StdInputData.SetCellString("USTimeInStr", curTimeStepInt,
			usTimeInStr+usTimeIn.Unpack().CoordsString()+usTimeIn2Str)
	}
}

func (ev *PVLVEnv) IsTestTrial(ti *data.TrialInstance) bool {
	testFlag := ti.TestFlag
	eTrlNm := ti.TrialName
	// testing both is an extra safety measure for congruence
	if testFlag && strings.Contains(strings.ToLower(eTrlNm), "_test") {
		return true
	} else {
		if testFlag || strings.Contains(strings.ToLower(eTrlNm), "_test") {
			fmt.Printf("ERROR: TrialName (%s) and TestFlag (%v) seem to be incongruent!\n",
				eTrlNm, testFlag)
		}
		return false
	}
}
