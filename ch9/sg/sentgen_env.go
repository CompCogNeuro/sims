// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/esg"
	"github.com/emer/etable/etensor"
)

// SentGenEnv generates sentences using a grammar that is parsed from a
// text file.  The core of the grammar is rules with various items
// chosen at random during generation -- these items can be
// more rules terminal tokens.
type SentGenEnv struct {
	Nm          string            `desc:"name of this environment"`
	Dsc         string            `desc:"description of this environment"`
	Rules       esg.Rules         `desc:"core sent-gen rules -- loaded from a grammar / rules file -- Gen() here generates one sentence"`
	PPassive    float32           `desc:"probability of generating passive sentence forms"`
	PQueryPrior float32           `desc:"probability of querying prior role-filler info before end of sentence"`
	WordTrans   map[string]string `desc:"translate unambiguous words into ambiguous words"`
	Words       []string          `desc:"list of words used for activating state units according to index"`
	WordMap     map[string]int    `desc:"map of words onto index in Words list"`
	Roles       []string          `desc:"list of roles used for activating state units according to index"`
	RoleMap     map[string]int    `desc:"map of roles onto index in Roles list"`
	Fillers     []string          `desc:"list of filler concepts used for activating state units according to index"`
	FillerMap   map[string]int    `desc:"map of roles onto index in Words list"`
	CurSentOrig []string          `desc:"original current sentence as generated from Rules"`
	CurSent     []string          `desc:"current sentence, potentially transformed to passive form"`
	SentInputs  [][]string        `desc:"generated sequence of sentence inputs including role-filler queries"`
	SentIdx     env.CurPrvInt     `desc:"current index within sentence inputs"`
	SentState   etensor.Float32   `desc:"current sentence activation state"`
	RoleState   etensor.Float32   `desc:"current role query activation state"`
	FillerState etensor.Float32   `desc:"current filler query activation state"`
	Run         env.Ctr           `view:"inline" desc:"current run of model as provided during Init"`
	Epoch       env.Ctr           `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Seq         env.Ctr           `view:"inline" desc:"sequence counter within epoch"`
	Trial       env.Ctr           `view:"inline" desc:"trial is the step counter within sequence - how many steps taken within current sequence -- it resets to 0 at start of each sequence"`
}

func (ev *SentGenEnv) Name() string { return ev.Nm }
func (ev *SentGenEnv) Desc() string { return ev.Dsc }

// InitTMat initializes matrix and labels to given size
func (ev *SentGenEnv) Validate() error {
	ev.Rules.Validate()
	return nil
}

func (ev *SentGenEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (ev *SentGenEnv) States() env.Elements {
	els := env.Elements{
		{"Input", []int{len(ev.Words)}, nil},
		{"Role", []int{len(ev.Roles)}, nil},
		{"Filler", []int{len(ev.Fillers)}, nil},
	}
	return els
}

func (ev *SentGenEnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.SentState
	case "Role":
		return &ev.RoleState
	case "Filler":
		return &ev.FillerState
	}
	return nil
}

func (ev *SentGenEnv) Actions() env.Elements {
	return nil
}

// String returns the current state as a string
func (ev *SentGenEnv) String() string {
	return ""
}

func (ev *SentGenEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Seq.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0

	ev.SentState.SetShape([]int{len(ev.Words)}, nil, []string{"Words"})
	ev.RoleState.SetShape([]int{len(ev.Roles)}, nil, []string{"Roles"})
	ev.FillerState.SetShape([]int{len(ev.Fillers)}, nil, []string{"Fillers"})
}

// NextSent generates the next sentence and all the queries for it
func (ev *SentGenEnv) NextSent() {
	ev.CurSent = ev.Rules.Gen()
	ev.Rules.TrimStateQualifiers()
	ev.SentIdx.Set(-1)
	ev.SentSeqActiveDet() // todo: passive
}

// TransWord gets the translated word
func (ev *SentGenEnv) TransWord(word string) string {
	if tr, has := ev.WordTrans[word]; has {
		return tr
	}
	return word
}

// AddInput adds a new input with given sentence index word and role query
func (ev *SentGenEnv) AddInput(sidx int, role string) {
	wrd := ev.TransWord(ev.CurSent[sidx])
	fil := ev.Rules.States[role]
	ev.SentInputs = append(ev.SentInputs, []string{wrd, role, fil})
}

// SentSeqActiveProb generates active-form sequence of inputs, probabilistic prior queries
func (ev *SentGenEnv) SentSeqActiveProb() {
	ev.SentInputs = make([][]string, 0, 50)
	mod := ev.Rules.States["Mod"]
	seq := []string{"Agent", "Action", "Patient"}
	for si, sq := range seq {
		ev.AddInput(si, sq)
		for ri := 0; ri < si; ri++ {
			if erand.BoolProb(float64(ev.PQueryPrior), -1) {
				ev.AddInput(si, seq[ri])
			}
		}
	}
	// get any modifier words with action query
	slen := len(ev.CurSent)
	for si := 3; si < slen-1; si++ {
		ri := rand.Intn(3) // choose a role to query at random
		ev.AddInput(si, seq[ri])
	}
	// last one has it all, always
	ev.AddInput(slen-1, mod)
	for _, sq := range seq {
		ev.AddInput(slen-1, sq)
	}
}

// SentSeqActiveDet generates active-form sequence of inputs, deterministic sequence
func (ev *SentGenEnv) SentSeqActiveDet() {
	ev.SentInputs = make([][]string, 0, 50)
	mod := ev.Rules.States["Mod"]
	seq := []string{"Agent", "Action", "Patient"}
	for si, sq := range seq {
		ev.AddInput(si, sq)
	}
	// get any modifier words with random query
	slen := len(ev.CurSent)
	for si := 3; si < slen-1; si++ {
		ri := rand.Intn(3) // choose a role to query at random
		ev.AddInput(si, seq[ri])
	}
	// last one has it all, always
	ev.AddInput(slen-1, mod)
	for _, sq := range seq {
		ev.AddInput(slen-1, sq)
	}
}

// RenderState renders the current state
func (ev *SentGenEnv) RenderState() {
	cur := ev.SentInputs[ev.SentIdx.Cur]
	widx := ev.WordMap[cur[0]]
	ev.SentState.SetZeros()
	ev.SentState.SetFloat1D(widx, 1)
	ridx := ev.RoleMap[cur[1]]
	ev.RoleState.SetZeros()
	ev.RoleState.SetFloat1D(ridx, 1)
	fidx := ev.FillerMap[cur[2]]
	ev.FillerState.SetZeros()
	ev.FillerState.SetFloat1D(fidx, 1)
}

// NextState generates the next inputs
func (ev *SentGenEnv) NextState() {
	if ev.SentIdx.Cur < 0 {
		ev.NextSent()
	}
	ev.SentIdx.Set(ev.SentIdx.Cur + 1)
	if ev.SentIdx.Cur >= len(ev.CurSent) {
		ev.NextSent()
	}
	ev.RenderState()
}

func (ev *SentGenEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	ev.NextState()
	ev.Trial.Incr()
	if ev.SentIdx.Cur == 0 {
		if ev.Seq.Incr() {
			ev.Epoch.Incr()
		}
	}
	return true
}

func (ev *SentGenEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *SentGenEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Sequence:
		return ev.Seq.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*SentGenEnv)(nil)
