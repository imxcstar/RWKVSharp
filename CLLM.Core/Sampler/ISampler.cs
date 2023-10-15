using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLLM.Core.Sampler
{
    public interface ISampler
    {
        public int Sample(IEnumerable<float> logits);
    }
}
