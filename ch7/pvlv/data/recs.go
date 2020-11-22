package data

import (
	"errors"
	"github.com/goki/ki/kit"
	"math/rand"
	"reflect"
)

type IRecs interface {
	Length() int
	Append(interface{}) IRecs
	Get(int) interface{}
}

type Recs struct {
	Records IRecs
	DType   reflect.Kind
	Index   []int
	INext   int
	NRead   int
	Order   DataLoopOrder
}

type DataLoopOrder int

const (
	SEQUENTIAL DataLoopOrder = iota
	PERMUTED
	RANDOM
	DataLoopOrderN
)

var KiT_DataLoopOrder = kit.Enums.AddEnum(DataLoopOrderN, kit.NotBitFlag, nil)

func IntSequence(begin, end, step int) (sequence []int) {
	if step == 0 {
		panic(errors.New("step must not be zero"))
	}
	count := 0
	if (end > begin && step > 0) || (end < begin && step < 0) {
		count = (end-step-begin)/step + 1
	}

	sequence = make([]int, count)
	for i := 0; i < count; i, begin = i+1, begin+step {
		sequence[i] = begin
	}
	return
}

func NewRecs(irecs IRecs) *Recs {
	recs := new(Recs)
	recs.Records = irecs
	recs.Sequential()
	recs.Reset()
	return recs
}

// Set to initial state just before reading
func (recs *Recs) Reset() {
	recs.NRead = 0
	recs.INext = -1
}

// Set the entire index array for the TrialInstanceRecs
func (recs *Recs) SetIndex(ix []int) error {
	if len(ix) > recs.Records.Length() {
		return errors.New("supplied index is longer than data")
	}
	if len(ix) > recs.Records.Length() {
		return errors.New("supplied index is longer than data")
	}
	recs.Index = ix
	recs.Reset()
	return nil
}

func (recs *Recs) SetPos(i int) {
	recs.INext = i - 1
}
func (recs *Recs) Length() int {
	return recs.Records.Length()
}
func (recs *Recs) Cur() int {
	return recs.INext + 1
}

func (recs *Recs) SetOrder(order DataLoopOrder) {
	recs.Order = order
	if order == SEQUENTIAL {
		recs.Sequential()
	} else if order == PERMUTED {
		recs.Permute()
	}
	recs.Reset()
}

func (recs *Recs) Permute() {
	recs.Index = rand.Perm(recs.Records.Length())
}
func (recs *Recs) Sequential() {
	recs.Index = IntSequence(0, recs.Records.Length(), 1)
}
func (recs *Recs) GetIndex() []int {
	return recs.Index
}

func (recs *Recs) AtEnd() bool {
	return recs.NRead >= recs.Records.Length()
}

func (recs *Recs) WriteNext(rec interface{}) {
	recs.Records = recs.Records.Append(rec)
	recs.Index = append(recs.Index, recs.Records.Length()-1)
}
func (recs *Recs) ReadNext() interface{} {
	recs.NRead++
	if recs.Order == RANDOM {
		return recs.Records.Get(rand.Intn(recs.Records.Length()))
	} else {
		recs.INext++
		return recs.Records.Get(recs.Index[recs.INext])
	}
}
