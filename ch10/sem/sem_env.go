// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/fs"
	"math/rand"
	"sort"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
)

// SemEnv presents paragraphs of text, loaded from file(s)
// This assumes files have all been pre-filtered so only relevant words are present.
type SemEnv struct {
	// name of this environment
	Name string

	// if true, go sequentially through paragraphs -- else permuted
	Sequential bool

	// permuted order of paras to present if not sequential -- updated every time through the list
	Order []int

	// paths to text files
	TextFiles []string

	// list of words, in alpha order
	Words []string

	// map of words onto index in Words list
	WordMap map[string]int

	// current para activation state
	CurParaState tensor.Float32

	// paragraphs
	Paras [][]string

	// special labels for each paragraph (provided in first word of para)
	ParaLabels []string

	// trial is the step counter within epoch -- this is the index into Paras
	Trial env.Counter `display:"inline"`
}

func (ev *SemEnv) Label() string { return ev.Name }

func (ev *SemEnv) State(element string) tensor.Tensor {
	switch element {
	case "Input":
		return &ev.CurParaState
	}
	return nil
}

func (ev *SemEnv) Init(run int) {
	ev.Trial.Scale = etime.Trial
	ev.Trial.Init()
	ev.InitOrder()

	nw := len(ev.Words)
	ev.CurParaState.SetShape([]int{nw})
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

// OpenTextsFS opens multiple text files from filesystem file.
// use this as main API even if only opening one text
func (ev *SemEnv) OpenTextsFS(fsys fs.FS, txts ...string) {
	ev.TextFiles = txts
	ev.Paras = make([][]string, 0, 2000)
	ev.ParaLabels = make([]string, 0, 2000)
	for _, tf := range ev.TextFiles {
		fb, err := fs.ReadFile(fsys, tf)
		if errors.Log(err) != nil {
			return
		}
		ev.ScanText(bytes.NewBuffer(fb))
	}
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

func (ev *SemEnv) OpenWordsFS(fsys fs.FS, fname string) {
	fp, err := fsys.Open(fname)
	defer fp.Close()
	if errors.Log(err) != nil {
		return
	}
	ev.ScanWords(fp)
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
	if ev.Trial.Incr() { // if true, hit max, reset to 0
		randx.PermuteInts(ev.Order)
	}
	ev.SetParaState()
	return true
}

func (ev *SemEnv) Action(element string, input tensor.Tensor) {
	// nop
}

// String returns the string rep of the LED env state
func (ev *SemEnv) String() string {
	cpar := ev.CurPara()

	if cpar == nil || len(cpar) == 0 {
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

// ParaIndex returns the current idx number in Paras, based on Sequential / perumuted Order
func (ev *SemEnv) ParaIndex() int {
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
	pidx := ev.ParaIndex()
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

// Compile-time check that implements Env interface
var _ env.Env = (*SemEnv)(nil)
