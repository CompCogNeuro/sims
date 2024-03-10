package data

import "github.com/emer/leabra/v2/pvlv"

// A set of trial groups, sourced from an TrialParams list, instantiated according to the
// PercentOfTotal field in the source list.
// this is what we get after calling SetActiveTrialList
// Still not fully instantiated, US is still a probability
type TrialInstance struct {
	TrialName            string
	ValenceContext       pvlv.Valence
	USFlag               bool
	TestFlag             bool
	MixedUS              bool
	USProb               float64
	USMagnitude          float64
	AlphaTicksPerTrialGp int
	CS                   string
	CSTimeStart          int
	CSTimeEnd            int
	CS2TimeStart         int
	CS2TimeEnd           int
	USTimeStart          int
	USTimeEnd            int
	Context              string
	USType               string
}

type TrialInstanceList []*TrialInstance

type TrialInstanceRecs struct {
	Recs
}

func NewTrialInstanceRecs(til *TrialInstanceList) *TrialInstanceRecs {
	if til == nil {
		til = new(TrialInstanceList)
	}
	recs := &TrialInstanceRecs{Recs: *NewRecs(til)}
	return recs
}

func (til *TrialInstanceList) Length() int {
	return len(*til)
}

func (til *TrialInstanceList) Append(ins interface{}) IRecs {
	ret := append(*til, ins.(*TrialInstance))
	return &ret
}

func (til *TrialInstanceList) Get(i int) interface{} {
	return (*til)[i]
}

func (til *TrialInstanceRecs) ReadNext() *TrialInstance {
	return til.Recs.ReadNext().(*TrialInstance)
}

var _ IRecs = (*TrialInstanceList)(nil) // check for interface implementation
