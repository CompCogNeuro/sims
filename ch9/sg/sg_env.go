// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"strings"

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
	Nm            string            `desc:"name of this environment"`
	Dsc           string            `desc:"description of this environment"`
	Rules         esg.Rules         `desc:"core sent-gen rules -- loaded from a grammar / rules file -- Gen() here generates one sentence"`
	PPassive      float32           `desc:"probability of generating passive sentence forms"`
	PQueryPrior   float32           `desc:"probability of querying prior role-filler info before end of sentence"`
	WordTrans     map[string]string `desc:"translate unambiguous words into ambiguous words"`
	Words         []string          `desc:"list of words used for activating state units according to index"`
	WordMap       map[string]int    `desc:"map of words onto index in Words list"`
	Roles         []string          `desc:"list of roles used for activating state units according to index"`
	RoleMap       map[string]int    `desc:"map of roles onto index in Roles list"`
	Fillers       []string          `desc:"list of filler concepts used for activating state units according to index"`
	FillerMap     map[string]int    `desc:"map of roles onto index in Words list"`
	AmbigVerbs    []string          `desc:"ambiguous verbs"`
	AmbigNouns    []string          `desc:"ambiguous nouns"`
	AmbigVerbsMap map[string]int    `desc:"map of ambiguous verbs"`
	AmbigNounsMap map[string]int    `desc:"map of ambiguous nouns"`
	CurSentOrig   []string          `desc:"original current sentence as generated from Rules"`
	CurSent       []string          `desc:"current sentence, potentially transformed to passive form"`
	NAmbigNouns   int               `desc:"number of ambiguous nouns"`
	NAmbigVerbs   int               `desc:"number of ambiguous verbs (0 or 1)"`
	SentInputs    [][]string        `desc:"generated sequence of sentence inputs including role-filler queries"`
	SentIdx       env.CurPrvInt     `desc:"current index within sentence inputs"`
	SentState     etensor.Float32   `desc:"current sentence activation state"`
	RoleState     etensor.Float32   `desc:"current role query activation state"`
	FillerState   etensor.Float32   `desc:"current filler query activation state"`
	Run           env.Ctr           `view:"inline" desc:"current run of model as provided during Init"`
	Epoch         env.Ctr           `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Seq           env.Ctr           `view:"inline" desc:"sequence counter within epoch"`
	Tick          env.Ctr           `view:"inline" desc:"tick counter within sequence"`
	Trial         env.Ctr           `view:"inline" desc:"trial is the step counter within sequence - how many steps taken within current sequence -- it resets to 0 at start of each sequence"`
}

func (ev *SentGenEnv) Name() string { return ev.Nm }
func (ev *SentGenEnv) Desc() string { return ev.Dsc }

// InitTMat initializes matrix and labels to given size
func (ev *SentGenEnv) Validate() error {
	ev.Rules.Validate()
	return nil
}

func (ev *SentGenEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Tick, env.Trial}
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

func (ev *SentGenEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Seq.Init()
	ev.Tick.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.SentIdx.Set(-1)

	ev.MapsFmWords()

	ev.SentState.SetShape([]int{len(ev.Words)}, nil, []string{"Words"})
	ev.RoleState.SetShape([]int{len(ev.Roles)}, nil, []string{"Roles"})
	ev.FillerState.SetShape([]int{len(ev.Fillers)}, nil, []string{"Fillers"})
}

func (ev *SentGenEnv) MapsFmWords() {
	ev.WordMap = make(map[string]int, len(ev.Words))
	for i, wrd := range ev.Words {
		ev.WordMap[wrd] = i
	}
	ev.RoleMap = make(map[string]int, len(ev.Roles))
	for i, wrd := range ev.Roles {
		ev.RoleMap[wrd] = i
	}
	ev.FillerMap = make(map[string]int, len(ev.Fillers))
	for i, wrd := range ev.Fillers {
		ev.FillerMap[wrd] = i
	}
	ev.AmbigVerbsMap = make(map[string]int, len(ev.AmbigVerbs))
	for i, wrd := range ev.AmbigVerbs {
		ev.AmbigVerbsMap[wrd] = i
	}
	ev.AmbigNounsMap = make(map[string]int, len(ev.AmbigNouns))
	for i, wrd := range ev.AmbigNouns {
		ev.AmbigNounsMap[wrd] = i
	}
}

// CurInputs returns current inputs triple from SentInputs
func (ev *SentGenEnv) CurInputs() []string {
	if ev.SentIdx.Cur >= 0 && ev.SentIdx.Cur < len(ev.SentInputs) {
		return ev.SentInputs[ev.SentIdx.Cur]
	}
	return nil
}

// String returns the current state as a string
func (ev *SentGenEnv) String() string {
	cur := ev.CurInputs()
	if cur != nil {
		return fmt.Sprintf("%s %s=%s", cur[0], cur[1], cur[2])
	}
	return ""
}

// NextSent generates the next sentence and all the queries for it
func (ev *SentGenEnv) NextSent() {
	// ev.Rules.Trace = true
	ev.CurSent = ev.Rules.Gen()
	// fmt.Printf("%v\n", ev.CurSent)
	ev.Rules.TrimStateQualifiers()
	ev.SentStats()
	ev.SentIdx.Set(0)
	// ev.SentSeqActiveDet() // todo: passive
	// ev.SentSeqActiveDetNoSum() // todo: passive
	ev.SentSeqActiveDetRndSum() // todo: passive
}

// TransWord gets the translated word
func (ev *SentGenEnv) TransWord(word string) string {
	word = strings.ToLower(word)
	if tr, has := ev.WordTrans[word]; has {
		return tr
	}
	return word
}

// SentStats computes stats on sentence (ambig words)
func (ev *SentGenEnv) SentStats() {
	ev.NAmbigNouns = 0
	ev.NAmbigVerbs = 0
	for _, wrd := range ev.CurSent {
		wrd = ev.TransWord(wrd)
		if _, has := ev.AmbigVerbsMap[wrd]; has {
			ev.NAmbigVerbs++
		}
		if _, has := ev.AmbigNounsMap[wrd]; has {
			ev.NAmbigNouns++
		}
	}
}

// CheckWords reports errors if words not found, if not empty
func (ev *SentGenEnv) CheckWords(wrd, role, fill string) []error {
	var errs []error
	if _, ok := ev.WordMap[wrd]; !ok {
		errs = append(errs, fmt.Errorf("word not found in WordMap: %s, sent: %v", wrd, ev.CurSent))
	}
	if _, ok := ev.RoleMap[role]; !ok {
		errs = append(errs, fmt.Errorf("word not found in RoleMap: %s, sent: %v", role, ev.CurSent))
	}
	if _, ok := ev.FillerMap[fill]; !ok {
		errs = append(errs, fmt.Errorf("word not found in FillerMap: %s, sent: %v", fill, ev.CurSent))
	}
	if errs != nil {
		for _, err := range errs {
			fmt.Println(err)
		}
	}
	return errs
}

// AddInput adds a new input with given sentence index word and role query
func (ev *SentGenEnv) AddInput(sidx int, role string) {
	wrd := ev.TransWord(ev.CurSent[sidx])
	fill := ev.Rules.States[role]
	ev.CheckWords(wrd, role, fill)
	ev.SentInputs = append(ev.SentInputs, []string{wrd, role, fill})
}

// AddQuestion adds a new input with 'question' word and role query
func (ev *SentGenEnv) AddQuestion(role string) {
	wrd := "question"
	fill := ev.Rules.States[role]
	ev.CheckWords(wrd, role, fill)
	ev.SentInputs = append(ev.SentInputs, []string{wrd, role, fill})
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

// SentSeqActiveDetNoSum generates active-form sequence of inputs, deterministic sequence, no summary
func (ev *SentGenEnv) SentSeqActiveDetNoSum() {
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
	ev.AddInput(slen-1, mod)
	// last one has it all, always
	// for _, sq := range seq {
	// 	ev.AddInput(slen-1, sq)
	// }
}

// SentSeqActiveDetRndSum one random summary question
func (ev *SentGenEnv) SentSeqActiveDetRndSum() {
	ev.SentInputs = make([][]string, 0, 50)
	mod := ev.Rules.States["Mod"]
	seq := []string{"Agent", "Action", "Patient", mod}
	for si, sq := range seq {
		ev.AddInput(si, sq)
	}
	// get any modifier words with random query
	slen := len(ev.CurSent)
	for si := 3; si < slen-1; si++ {
		ri := rand.Intn(3) // choose a role to query at random
		ev.AddInput(si, seq[ri])
	}
	ev.AddInput(slen-1, mod)
	ri := rand.Intn(3) // choose a role to query at random
	ev.AddQuestion(seq[ri])
}

// RenderState renders the current state
func (ev *SentGenEnv) RenderState() {
	ev.SentState.SetZeros()
	ev.RoleState.SetZeros()
	ev.FillerState.SetZeros()
	cur := ev.CurInputs()
	if cur == nil {
		return
	}
	widx := ev.WordMap[cur[0]]
	ev.SentState.SetFloat1D(widx, 1)
	ridx := ev.RoleMap[cur[1]]
	ev.RoleState.SetFloat1D(ridx, 1)
	fidx := ev.FillerMap[cur[2]]
	ev.FillerState.SetFloat1D(fidx, 1)
}

// NextState generates the next inputs
func (ev *SentGenEnv) NextState() {
	if ev.SentIdx.Cur < 0 {
		ev.NextSent()
	} else {
		ev.SentIdx.Incr()
	}
	if ev.SentIdx.Cur >= len(ev.SentInputs) {
		ev.NextSent()
	}
	ev.RenderState()
}

func (ev *SentGenEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	ev.NextState()
	ev.Trial.Incr()
	ev.Tick.Incr()
	if ev.SentIdx.Cur == 0 {
		ev.Tick.Init()
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
	case env.Tick:
		return ev.Tick.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*SentGenEnv)(nil)
