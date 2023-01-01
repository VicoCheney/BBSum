from argparse import ArgumentParser
import os
from metrics import evaluate
from training_loop import train


def main_parser(parser):

    parser.add_argument("--save_dir", type=str, default=os.path.join(os.getcwd(), 'save_dir'), help="saving models")
    parser.add_argument("--tmp_dir", type=str, default=os.path.join(os.getcwd(), 'tmp_dir'), help="saving ddp tmp files")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.getcwd(), 'log_dir'), help="saving logs")
    parser.add_argument("--num_epochs", type=int, default=10, help="num epoch")
    parser.add_argument('--compresser_model', type=str, default='roberta-base', help='name of pretrained models')
    parser.add_argument('--summarizer_model', type=str, default='facebook/bart-large', help='name of pretrained models')
    parser.add_argument('--version', type=int, default=0, help='the version to save or restore')
    parser.add_argument('--step_size', type=int, default=20000, help='the version to save or restore')
    parser.add_argument('--num_samples', type=str, default='5,5', help='num of continous, discrete random samples and promising samples')
    parser.add_argument('--times', type=str, default='3,3', help='memreplay times')
    parser.add_argument('--batch_size_inference', type=int, default=32, help='batch_size in memreplay')
    parser.add_argument("--gpus", type=int, default=0, help="available gpus")
    parser.add_argument('--train_source', type=str, default=os.path.join(os.getcwd(), 'data', 'govreport_train.pkl'), help='training dataset')
    parser.add_argument('--test_source', type=str, default=os.path.join(os.getcwd(), 'data', 'govreport_test.pkl'), help='test dataset')
    parser.add_argument('--validation_source', type=str, default=os.path.join(os.getcwd(), 'data', 'govreport_validation.pkl'), help='validation dataset')
    parser.add_argument('--only_train', action='store_true')
    parser.add_argument('--only_evaluate', action='store_true')
    parser.add_argument('--lr1', type=float, default=5e-5, help='learning rate of introspector')
    parser.add_argument('--weight_decay1', type=float, default=0, help='weight decay of introspector')
    parser.add_argument('--bert_batch_size', type=int, default=32, help='gradient batch_size')
    parser.add_argument('--lr2', type=float, default=5e-5, help='learning rate of reasoner')
    parser.add_argument('--weight_decay2', type=float, default=0, help='weight decay of reasoner')
    parser.add_argument('--bart_batch_size', type=int, default=2, help='gradient batch_size')
    parser.add_argument('--levelup_threshold', type=float, default=0.1, help='gradient batch_size')
    parser.add_argument('--leveldown_threshold', type=float, default=-0.05, help='gradient batch_size')

    return parser


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser = main_parser(parser)
    config = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not config.only_evaluate:
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Training Started!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        train(config)
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Training Finished!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")

    if not config.only_train:
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Evaluation Started!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        evaluate(config, mode='test')
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Evaluation Finished!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")