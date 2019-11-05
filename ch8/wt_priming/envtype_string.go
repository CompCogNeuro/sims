// Code generated by "stringer -type=EnvType"; DO NOT EDIT.

package main

import (
	"errors"
	"strconv"
)

var _ = errors.New("dummy error")

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[TrainB-0]
	_ = x[TrainA-1]
	_ = x[TrainAll-2]
	_ = x[TestA-3]
	_ = x[TestB-4]
	_ = x[TestAll-5]
	_ = x[EnvTypeN-6]
}

const _EnvType_name = "TrainBTrainATrainAllTestATestBTestAllEnvTypeN"

var _EnvType_index = [...]uint8{0, 6, 12, 20, 25, 30, 37, 45}

func (i EnvType) String() string {
	if i < 0 || i >= EnvType(len(_EnvType_index)-1) {
		return "EnvType(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _EnvType_name[_EnvType_index[i]:_EnvType_index[i+1]]
}

func (i *EnvType) FromString(s string) error {
	for j := 0; j < len(_EnvType_index)-1; j++ {
		if s == _EnvType_name[_EnvType_index[j]:_EnvType_index[j+1]] {
			*i = EnvType(j)
			return nil
		}
	}
	return errors.New("String: " + s + " is not a valid option for type: EnvType")
}
