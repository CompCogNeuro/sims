// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/etable/v2/etensor"
)

// SemEnv presents paragraphs of text, loaded from file(s)
// This assumes files have all been pre-filtered so only relevant words are present.
type SemEnv struct {
	Nm           string          `desc:"name of this environment"`
	Dsc          string          `desc:"description of this environment"`
	Sequential   bool            `desc:"if true, go sequentially through paragraphs -- else permuted"`
	Order        []int           `desc:"permuted order of paras to present if not sequential -- updated every time through the list"`
	TextFiles    []string        `desc:"paths to text files"`
	Words        []string        `desc:"list of words, in alpha order"`
	WordMap      map[string]int  `desc:"map of words onto index in Words list"`
	CurParaState etensor.Float32 `desc:"current para activation state"`
	Paras        [][]string      `desc:"paragraphs"`
	ParaLabels   []string        `desc:"special labels for each paragraph (provided in first word of para)"`
	Run          env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch        env.Ctr         `view:"inline" desc:"number of times through Seq.Max number of sequences"`
	Trial        env.Ctr         `view:"inline" desc:"trial is the step counter within epoch -- this is the index into Paras"`
}

func (ev *SemEnv) Name() string { return ev.Nm }
func (ev *SemEnv) Desc() string { return ev.Dsc }

func (ev *SemEnv) Validate() error {
	return nil
}

func (ev *SemEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Sequence, env.Trial}
}

func (ev *SemEnv) States() env.Elements {
	sz := ev.CurParaState.Shapes()
	nms := ev.CurParaState.DimNames()
	els := env.Elements{
		{"Input", sz, nms},
	}
	return els
}

func (ev *SemEnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.CurParaState
	}
	return nil
}

func (ev *SemEnv) Actions() env.Elements {
	return nil
}

func (ev *SemEnv) Defaults() {
}

func (ev *SemEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.InitOrder()

	nw := len(ev.Words)
	ev.CurParaState.SetShape([]int{nw}, nil, []string{"Words"})
}

// InitOrder initializes the order based on current Paras, resets Trial.Cur = -1 too
func (ev *SemEnv) InitOrder() {
	np := len(ev.Paras)
	ev.Order = rand.Perm(np) // always start with new one so random order is identical
	// and always maintain Order so random number usage is same regardless, and if
	// user switches between Sequential and random at any point, it all works..
	ev.Trial.Max = np
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

// OpenTexts opens multiple text files -- use this as main API
// even if only opening one text
func (ev *SemEnv) OpenTexts(txts []string) {
	ev.TextFiles = txts
	ev.Paras = make([][]string, 0, 2000)
	for _, tf := range ev.TextFiles {
		ev.OpenText(tf)
	}
}

// OpenText opens one text file
func (ev *SemEnv) OpenText(fname string) {
	fp, err := os.Open(fname)
	defer fp.Close()
	if err != nil {
		log.Println(err)
		return
	}
	ev.ScanText(fp)
}

// ScanText scans given text file from reader, adding to Paras
func (ev *SemEnv) ScanText(fp io.Reader) {
	scan := bufio.NewScanner(fp) // line at a time
	cur := []string{}
	lbl := ""
	for scan.Scan() {
		b := scan.Bytes()
		bs := string(b)
		sp := strings.Fields(bs)
		if len(sp) == 0 {
			ev.Paras = append(ev.Paras, cur)
			ev.ParaLabels = append(ev.ParaLabels, lbl)
			cur = []string{}
			lbl = ""
		} else {
			coli := strings.Index(sp[0], ":")
			if coli > 0 {
				lbl = sp[0][:coli]
				sp = sp[1:]
			}
			cur = append(cur, sp...)
		}
	}
	if len(cur) > 0 {
		ev.Paras = append(ev.Paras, cur)
		ev.ParaLabels = append(ev.ParaLabels, lbl)
	}
}

// OpenTextsAsset opens multiple text files from bindata Asset.
// use this as main API even if only opening one text
func (ev *SemEnv) OpenTextsAsset(txts []string) {
	ev.TextFiles = txts
	ev.Paras = make([][]string, 0, 2000)
	ev.ParaLabels = make([]string, 0, 2000)
	for _, tf := range ev.TextFiles {
		ev.OpenTextAsset(tf)
	}
}

// OpenTextAsset opens one text file from bindata.Asset
func (ev *SemEnv) OpenTextAsset(fname string) {
	fp, err := Asset(fname)
	if err != nil {
		log.Println(err)
		return
	}
	ev.ScanText(bytes.NewBuffer(fp))
}

// CheckWords checks that the words in the slice (one word per index) are in the list.
// Returns error for any missing words.
func (ev *SemEnv) CheckWords(wrds []string) error {
	missing := ""
	for _, wrd := range wrds {
		_, ok := ev.WordMap[wrd]
		if !ok {
			missing += wrd + " "
		}
	}
	if missing != "" {
		return fmt.Errorf("CheckWords: these words were not found: %s", missing)
	}
	return nil
}

// SetParas sets the paragraphs from list of space-separated word strings -- each string is a paragraph.
// calls InitOrder after to reset.
// returns error for any missing words (from CheckWords)
func (ev *SemEnv) SetParas(paras []string) error {
	ev.Paras = make([][]string, len(paras))
	ev.ParaLabels = make([]string, len(paras))
	var err error
	for i, ps := range paras {
		lbl := ""
		sp := strings.Fields(ps)
		if len(sp) > 0 {
			coli := strings.Index(sp[0], ":")
			if coli > 0 {
				lbl = sp[0][:coli]
				sp = sp[1:]
			}
		}
		ev.Paras[i] = sp
		ev.ParaLabels[i] = lbl
		er := ev.CheckWords(sp)
		if er != nil {
			if err != nil {
				err = errors.New(err.Error() + " " + er.Error())
			} else {
				err = er
			}
		}
	}
	ev.InitOrder()
	return err
}

func (ev *SemEnv) OpenWords(fname string) {
	fp, err := os.Open(fname)
	defer fp.Close()
	if err != nil {
		log.Println(err)
		return
	}
	ev.ScanWords(fp)
}

// OpenWordsAsset opens one words file from bindata.Asset
func (ev *SemEnv) OpenWordsAsset(fname string) {
	fp, err := Asset(fname)
	if err != nil {
		log.Println(err)
		return
	}
	ev.ScanWords(bytes.NewBuffer(fp))
}

func (ev *SemEnv) ScanWords(fp io.Reader) {
	ev.Words = make([]string, 0, 3000)
	scan := bufio.NewScanner(fp) // line at a time
	for scan.Scan() {
		b := scan.Bytes()
		bs := string(b)
		sp := strings.Fields(bs)
		if len(sp) > 0 {
			ev.Words = append(ev.Words, sp...)
		}
	}
	ev.WordMapFmWords()
}

func (ev *SemEnv) WordMapFmWords() {
	ev.WordMap = make(map[string]int, len(ev.Words))
	for i, wrd := range ev.Words {
		ev.WordMap[wrd] = i
	}
}

func (ev *SemEnv) WordsFmWordMap() {
	ev.Words = make([]string, len(ev.WordMap))
	ctr := 0
	for wrd, _ := range ev.WordMap {
		ev.Words[ctr] = wrd
		ctr++
	}
	sort.Strings(ev.Words)
	for i, wrd := range ev.Words {
		ev.WordMap[wrd] = i
	}
}

func (ev *SemEnv) WordsFmText() {
	ev.WordMap = make(map[string]int, len(ev.Words))
	for _, para := range ev.Paras {
		for _, wrd := range para {
			ev.WordMap[wrd] = -1
		}
	}
	ev.WordsFmWordMap()
}

func (ev *SemEnv) Step() bool {
	ev.Epoch.Same()      // good idea to just reset all non-inner-most counters at start
	if ev.Trial.Incr() { // if true, hit max, reset to 0
		erand.PermuteInts(ev.Order)
		ev.Epoch.Incr()
	}
	ev.SetParaState()
	return true
}

func (ev *SemEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *SemEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*SemEnv)(nil)

// String returns the string rep of the LED env state
func (ev *SemEnv) String() string {
	cpar := ev.CurPara()
	if cpar == nil {
		return ""
	}
	str := cpar[0]
	if len(cpar) > 1 {
		str += " " + cpar[1]
		if len(cpar) > 2 {
			str += " ... " + cpar[len(cpar)-1]
		}
	}
	return str
}

// ParaIdx returns the current idx number in Paras, based on Sequential / perumuted Order
func (ev *SemEnv) ParaIdx() int {
	if ev.Trial.Cur < 0 {
		return -1
	}
	if ev.Sequential {
		return ev.Trial.Cur
	}
	return ev.Order[ev.Trial.Cur]
}

// CurPara returns the current paragraph
func (ev *SemEnv) CurPara() []string {
	pidx := ev.ParaIdx()
	if pidx >= 0 && pidx < len(ev.Paras) {
		return ev.Paras[pidx]
	}
	return nil
}

// SetParaState sets the para state from current para
func (ev *SemEnv) SetParaState() {
	cpar := ev.CurPara()
	ev.CurParaState.SetZeros()
	for _, wrd := range cpar {
		widx := ev.WordMap[wrd]
		ev.CurParaState.SetFloat1D(widx, 1)
	}
}
