#include "ArmorPredictor.hpp"

namespace helios_cv {

ArmorPredictor::ArmorPredictor(const APParams& params) {
    
}


ArmorPredictor::~ArmorPredictor() {
    kalman_filter_.reset();

}

} // helios_cv