
import os
import logging

import torch
import numpy as np

from transvae.transformer_models import TransVAE
from transvae.tvae_util import *

from sklearn.decomposition import PCA, KernelPCA

class GenerativeModelWrapper():
    """
    Wrapper class for generative model that must include:
        1. load_generative_model() method that loads the generative model based on the provided parameters
        2. encode_sequences_to_latent() method that encodes sequences to the (possibly reduced) latent space
        3. decode_latent_point() method that takes in a candidate latent point 
            from a (possibly reduced) latent space and decodes it to sequence space
            using the generative model's decoder. 
            If dimensionality reduction is used, this method should also handle the inverse projection
            from the reduced latent space back to the original latent space before decoding.
    
    Parameters:
    -----------
    params : dict
        A dictionary containing the parameters for the generative model, including:
            - model_type: str, the type of generative model
            - ckpt_fpath: str, the file path to the model checkpoint
            - char_dict_fpath: str, 
                the file path to the character dictionary
            - encoded_train_data_fpath: str, 
                the file path to the encoded training data 
                (used for fitting the dimensionality reducer if applicable)
            - dimensionality_reduction: dict, 
                containing the method and parameters for dimensionality reduction, 
                must include 'method' from ['pca', 'identity'], e.g.:
                {
                    "method": "pca",
                    "n_components": 5
                }
    bo_params : dict
        A dictionary containing the parameters for Bayesian optimization,
        which may be needed for some parameters of the generative model
        (e.g. batch size for decoding)
    """
    def __init__(self, params, bo_params):
        
        self.params = params
        self.bo_params = bo_params
        
        # check if dimensionality reduction method is valid
        self.params["dimensionality_reduction"]["method"] = self.params["dimensionality_reduction"].get("method", "identity").lower()
        if self.params["dimensionality_reduction"]["method"] not in ["pca", "kernel_pca", "identity"]:
            raise ValueError(f"Invalid dimensionality reduction method: {self.params['dimensionality_reduction']['method']}. Must be one of ['pca', 'kernel_pca', 'identity'].")

        # check if model files exist
        if not os.path.isfile(self.params['ckpt_fpath']):
            raise FileNotFoundError(f"Model checkpoint file not found at {self.params['ckpt_fpath']}")
        if not os.path.isfile(self.params['encoded_train_data_fpath']):
            raise FileNotFoundError(f"Encoded training data file not found at {self.params['encoded_train_data_fpath']}")
        if not os.path.isfile(self.params['char_dict_fpath']):
            raise FileNotFoundError(f"Character dictionary file not found at {self.params['char_dict_fpath']}")


        self.generative_model = self.load_generative_model()
        
        self.char_dict = self.generative_model.params['char_dict']
        
        if self.params['dimensionality_reduction']['method'] == "pca":
            self.dimensionality_reducer = PCA(
                n_components=self.params['dimensionality_reduction']['n_components']
            )

            # fit to encoded training data
            if self.params["encoded_train_data_fpath"].endswith(".pt"):
                _encoded_train_data = torch.load(self.params['encoded_train_data_fpath'])
            elif self.params["encoded_train_data_fpath"].endswith(".npy"):
                _encoded_train_data = np.load(self.params['encoded_train_data_fpath'])
                _encoded_train_data = torch.from_numpy(_encoded_train_data)

            self.dimensionality_reducer.fit(_encoded_train_data)

        elif self.params['dimensionality_reduction']['method'] == "kernel_pca":
            raise NotImplementedError("Kernel PCA dimensionality reduction is not yet implemented. Please use PCA or set dimensionality_reduction to None.")
            self.dimensionality_reducer = KernelPCA(n_components=self.params['dimensionality_reduction']['n_components'], kernel='rbf')
        elif self.params['dimensionality_reduction']['method'] == "identity":
            self.dimensionality_reducer = IdentityPCA(n_components=self.generative_model.params['d_latent'])

    def load_generative_model(self):
        """
        Loads the generative model based on the provided parameters.

        Returns:
        --------
        model : TransVAE (in this case)
            The loaded generative model.
        """

        model_src = self.params['ckpt_fpath']
        model_obj=torch.load(model_src, map_location=torch.device("cpu"))
        model = TransVAE(load_fn=model_src, workaround="cpu")
        
        model.params['HARDWARE']= 'cpu'
        model.params["BATCH_SIZE"] = self.bo_params["N_INITIALIZATION_POINTS"]

        model.model.eval()

        return model

    def encode_sequences_to_latent(self, sequences:list[str], return_latent_points=False) -> np.ndarray:
        """
        Encodes a list of sequences to the (possibly reduced) latent space.

        Parameters
        ----------
        sequences : list[str]
            A list of sequences to be encoded.
        return_latent_points : bool, optional
            If True, also returns the latent points in the original 
            latent space before dimensionality reduction.
            
        Returns
        -------
        np.ndarray
            A 2D array of shape (n_sequences, reduced_dim) representing the encoded sequences in the reduced latent space.
        """
        # encode sequences to original latent space
        # encoded_seqs = [self.encode_seq(seq) for seq in sequences]
        encoded_seqs = [seq for seq in sequences]
        encoded_seqs = torch.tensor(encoded_seqs)
        if len(encoded_seqs.shape) == 1:
            encoded_seqs = encoded_seqs.unsqueeze(0)
        
        with torch.no_grad():
            # latent_points = self.generative_model.encode(encoded_seqs)
            z, mu, logvar = self.generative_model.calc_mems(encoded_seqs.to_numpy(), log=False, save=False)
        latent_points = mu.cpu().numpy()

        # project from original latent space to reduced latent space
        latent_points_reduced = self.dimensionality_reducer.transform(latent_points)

        if return_latent_points:
            return latent_points_reduced, latent_points
        
        return latent_points_reduced

    def decode_seq(self, encoded_seq:list[int]) -> str:
            """
            Decodes an encoded sequence to a list of characters.

            Parameters
            ----------
            encoded_seq : list[int]
                A list of integers representing a sequence.

            Returns
            -------
            str
                The string representation of the sequence.
            """
            itos = {v:k for k,v in self.char_dict.items()}
            output = "".join([itos[i] for i in encoded_seq])
            output = output.strip("_")
            output = output.strip("<start>")
            output = output.strip("<end>")
            return output

    def encode_seq(self, sequence:str) -> list[int]:
        """
        Encodes a sequence to a list of integers.

        Parameters
        ----------
        sequence : str
            A sequence of characters.

        Returns
        -------
        list[int]
            A list of integers representing the sequence.
        """
        stoi = self.char_dict
        output = [stoi[i] for i in sequence]
        return output

    def decode_latent_point(self, latent_point_candidates, verbose=False):
        """
        Given latent point candidate(s) in the reduced space, 
        this function:
            1. inverse projects the candidate(s) back to the 
                original latent space using the dimensionality reducer
            2. decodes the candidate(s) from the original latent space 
                to sequence space using the generative model's decoder

        Parameters
        ----------
        latent_point_candidates : np.ndarray or torch.Tensor
            A 2D array of shape (n_candidates, reduced_dim) representing
            the candidate latent points in the reduced space.
        verbose : bool, optional
            If True, prints the candidate points in both reduced and original spaces.
        
        Returns
        -------
        candidate_sequences : list[str]
            A list of decoded sequences corresponding to the input candidate latent points.
        """

        # inverse projection from reduced space to original latent space
        # if the dimensionality reducer is an identity function,
        # this will simply return the input candidates without modification
        candidates_invproj = self.dimensionality_reducer.inverse_transform(latent_point_candidates)
            
        if verbose:
            print(f"candidate in reduced space: {latent_point_candidates}")
            print(f"candidate in original space: {candidates_invproj}")

        if isinstance(candidates_invproj, np.ndarray):
            candidates_invproj = torch.from_numpy(candidates_invproj)

        candidates_invproj = candidates_invproj.reshape(-1, candidates_invproj.shape[-1]) # (n_restarts, 1, d_latent) -> (n_restarts, d_latent)

        # decode from latent space --> sequence space
        with torch.no_grad():
            _decode_inv_proj = self.generative_model.greedy_decode(candidates_invproj)
            candidate_decoded = _decode_inv_proj 

        candidate_sequences = candidate_decoded

        return candidate_sequences


class IdentityPCA():
    def __init__(self, n_components):
        self.n_components = n_components
    def fit(self, X):
        return self
    def transform(self, X):
        return X
    def inverse_transform(self, X):
        return X