"""
Created on Sun Nov 28 18:27:32 2021

@author: m.benammar
for academic purposes only 
"""

from matplotlib import pyplot 
import numpy as np 
from PIL import Image as im
import jpeg_functions as fc 
import huffman_functions as hj 
import time
import math 

# Quantization matrix 
Q =100; # quality  factor of JPEG in %

Q_matrix = fc.quantization_matrix(Q) # Quantization matrix 

dc_time_Tree = []
dc_time_compress = []
dc_time_decompress = []
dc_avg_lens = []
dc_entropy = []

ac_time_Tree = []
ac_time_compress = []
ac_time_decompress = []
ac_avg_lens = []
ac_entropy = []

# Open the image (obtain an RGB image)
image_origin = im.open("ISAE_Logo_SEIS_clr.PNG")  
image_origin_eval = np.array(image_origin)

# Transform into YCrBR 
image_ycbcr = image_origin.convert('YCbCr') 
image_ycbcr_eval= np.array(image_ycbcr)
 
#  Troncate the image (to make simulations faster)
image_trunc = image_ycbcr_eval[644:724,624:704] 
# pyplot.imshow(image_trunc) 

# Initializations
n_row = np.size(image_trunc,0) # Number of rows 
n_col = np.size(image_trunc,1)  # Number of columns
N_blocks =  (np.round(n_row/8 * n_col/8 )).astype(np.int32) # Number of (8x8) blocks 

image_plane_rec = np.zeros((n_row,n_col,3),dtype=np.uint8) 
image_DC = np.zeros((N_blocks.astype(np.int32),3), dtype = np.int32)
image_DC_rec = np.zeros((N_blocks.astype(np.int32),3), dtype = np.int32)
image_AC = np.zeros((63*N_blocks,3),dtype = np.int32 )
image_AC_rec = np.zeros((63*N_blocks,3),dtype = np.int32 )
bits_per_pixel=np.zeros( 3 )
# ---------------------------------------------------------------
#                          Compression 
#  --------------------------------------------------------------

def calc_dc_avg_length(alph_dict: dict):
    avg_len: float = 0
    for cat, prob in alph_dict.items():
        avg_len += cat * prob
        
    return avg_len

def calc_ac_avg_length(encoding_dict: dict, alph_dict: dict):
    avg_len: float = 0
    for tup in encoding_dict.keys():
        avg_len += len(encoding_dict[tup]) * alph_dict[tup]
        
    return avg_len

def calc_entropy(alph_dict: dict):
    entropy: float = 0
    
    for prob in alph_dict.values():
        entropy -=prob * math.log2(prob)
    return entropy

for i_plane in range(0,3): 
    
    # Select a plane  Y, Cb, or Cr 
    image_plane = image_trunc[:,:,i_plane]

    for i_row in range(0,n_row,8): 
        for j_col in range(0,n_col,8):
            
            #  Compute the block number
            block_number =( np.round(i_row/8*(n_row/8)+ j_col/8 )).astype(np.int32)
 
            # Select of block of 8 by 8 pixels
            image_plane_block = image_plane[i_row:i_row+8, j_col:j_col+8]     
            
            # DCT transform
            image_dct = fc.DCT(image_plane_block)
            
            # Quantization
            image_dct_quant = np.round(image_dct/Q_matrix ).astype(np.int32)
            
            # Zigzag reading  
            image_dct_quant_zgzg =  fc.zigzag(image_dct_quant)            
        
            # Separate DC and AC components    
            image_DC[block_number,i_plane] = image_dct_quant_zgzg[0]  
            image_AC[block_number*63 + 1:(block_number+1)*63,i_plane] = image_dct_quant_zgzg[1:63]
    
    # ---- DC components processing    
    # DPCM coding over DC components 
    image_DC_DPCM = image_DC[1:N_blocks,i_plane] - image_DC[0:N_blocks-1,i_plane]
    image_DC_0 = image_DC[0,i_plane] # first DC constant (not compressed)
    
    #  Map the resulting values in categoeries and amplitudes
    image_DC_DPCM_cat = fc.DC_category_vect(image_DC_DPCM) 
    image_DC_DPCM_amp = fc.DC_amplitude_vect(image_DC_DPCM) 
    list_image_cat_DC = np.ndarray.tolist(image_DC_DPCM_cat)  # create a list of DC components
    
    
    # --------------------------Students work on DC components --------------------------------- 
    # Compress with Huffman
    dc_cat_set = list(set(list_image_cat_DC))
    [dc_alph, num_chars] = hj.dict_freq_numbers(list_image_cat_DC, dc_cat_set)
    
    dc_avg_lens.append(calc_dc_avg_length(dc_alph))
    dc_entropy.append(calc_entropy(dc_alph))
    
    timein = time.time()
    dc_huff_tree = hj.build_huffman_tree(dc_alph)
    timefin = time.time()
    dc_time_Tree.append(timefin-timein)
    
    dc_encoding_dict = hj.generate_code(dc_huff_tree)
    
    
    timein = time.time()
    dc_compressed = hj.compress(text=list_image_cat_DC, encoding_dict=dc_encoding_dict)
    timefin = time.time()
    dc_time_compress.append(timefin-timein)
    
    
    # print(f"DC ENCODE: {dc_encoding_dict}")
    # print(f"DC TREE: {dc_huff_tree}")
    # print(f"DC COMPRESSED: {dc_compressed}")

    
    # Decompress with Huffman
    dc_decoding_dict = {v: k for k, v in dc_encoding_dict.items()}

    
    timein = time.time()
    decompressed_cat_DC = hj.decompress(bits=dc_compressed, decoding_dict=dc_decoding_dict)
    timefin = time.time()
    dc_time_decompress.append(timefin-timein)
    
    
    # print(f"DC DECODED: {decompressed_cat_DC}")
    print(list_image_cat_DC == decompressed_cat_DC)
   
    # ---------------------------------------------------------------------------------------------
    
    # ---- AC components processing    
    # RLE coding over AC components  
    AC_coeff = image_AC[:,i_plane] 
    [AC_coeff_rl, AC_coeff_amp]= fc.RLE(AC_coeff)
    list_image_rl_AC = np.ndarray.tolist(AC_coeff_rl)
    
    # --------------------------Students work on AC components ---------------------------------
    # Compress with Huffman
    ac_tuples = [tuple(x) for x in AC_coeff_rl]
    ac_unique_tuples = list(set(ac_tuples))
    
    [ac_alph, ac_num_chars] = hj.dict_freq_numbers_2(ac_tuples, ac_unique_tuples)
    
    
    timein = time.time()
    ac_huff_tree = hj.build_huffman_tree(ac_alph)
    timefin = time.time()
    ac_time_Tree.append(timefin-timein)
    
    ac_encoding_dict = hj.generate_code_2(ac_huff_tree)
    
    
    timein = time.time()
    ac_compressed = hj.compress_2(text=list_image_rl_AC, encoding_dict=ac_encoding_dict)
    timefin = time.time()
    ac_time_compress.append(timefin-timein)
    
    ac_avg_lens.append(calc_ac_avg_length(ac_encoding_dict, ac_alph))
    ac_entropy.append(calc_entropy(ac_alph))
    
    # print(f"AC TREE: {ac_huff_tree}")
    # print(f"AC ENCODING DICT: {ac_encoding_dict}")
    # print(f"AC: {ac_compressed}")
    
    
    # Decompress with Huffman
    ac_decoding_dict = {v: list(k) for k, v in ac_encoding_dict.items()}
    
    
    timein = time.time()
    decompressed_cat_AC = hj.decompress(bits=ac_compressed, decoding_dict=ac_decoding_dict)
    timefin = time.time()
    ac_time_decompress.append(timefin-timein)
    
    
    #print(f"AC DECODED: {decompressed_cat_AC}")
    
    print(list_image_rl_AC == decompressed_cat_AC)

    # --------------------------------Students work on the nb_bit/ pixel ---------------------
 
     
    
# ---------------------------------------------------------------
#                      Decompression 
#  --------------------------------------------------------------
    # ---- DC components processing  
    # Map the  categories and amplitudes DC components back into values  
    decompressed_cat_DC = np.array(decompressed_cat_DC) 
    image_DC_DPCM_rec = (fc.cat_ampl_to_DC_vect(decompressed_cat_DC,image_DC_DPCM_amp)) 
    
    # DPCM decoding of DC components 
    image_DC_rec[0,i_plane]=image_DC_0
    for i_block in range(1,N_blocks):
        image_DC_rec[i_block,i_plane]= image_DC_rec[i_block-1,i_plane] + image_DC_DPCM_rec[i_block-1]
        
    # ---- AC components processing      
    # Map AC components back into their original values  
    decompressed_cat_AC = np.array(decompressed_cat_AC)  
    image_AC_rec[:,i_plane] =  fc.RLE_inv(decompressed_cat_AC,AC_coeff_amp)

    for i_row in range(0,n_row,8): 
        for j_col in range(0,n_col,8):
            
            block_number = (np.round(i_row/8*(n_row/8)+ j_col/8 )).astype(np.int32)  

            # Combining AC and DC components 
            image_dct_quant_zgzg_rec = np.zeros(64, dtype= np.int32)
            image_dct_quant_zgzg_rec[1:63]= image_AC_rec[block_number*63 + 1:(block_number+1)*63,i_plane]
            image_dct_quant_zgzg_rec[0] = image_DC_rec[block_number,i_plane]
             
            # Inverse zigzag reading  
            image_dct_quant_rec = fc.zigzag_inv(image_dct_quant_zgzg_rec)
            
            # De-Quantization
            image_dct_rec = image_dct_quant_rec*Q_matrix 
            
            # Inverse DCT
            image_rec =  fc.DCT_inv(image_dct_rec) 
            image_plane_rec[i_row:i_row+8, j_col:j_col+8,i_plane ] = image_rec.astype(np.uint8)
            

# Recovering the image from the array of YCbCr
image_ycbcr_rec = im.fromarray(image_plane_rec,'YCbCr')

# Convert back to RGB
image_rec =  image_ycbcr_rec.convert('RGB')

# Plot the image 
pyplot.imshow(image_rec)
pyplot.show()

#The following part of the code aims at evaluating the performance of the Huffman Compression algoryhtm. 

print('As it can be seen in all the cases the compressed information is the same as the previous')

print('dc time to construct the tree',dc_time_Tree)
print('dc time to compress',dc_time_compress)
print('dc time to decompress',dc_time_decompress)
print(f"dc average lengths: {dc_avg_lens}")
print(f"dc entropies: {dc_entropy}")

print('ac time to construct the tree',ac_time_Tree)
print('ac time to compress',ac_time_compress)
print('ac time to decompress',ac_time_decompress)
print(f"ac average lengths: {ac_avg_lens}")
print(f"ac entropies: {ac_entropy}")

number_entries_dc = 0


print(sum(list_image_cat_DC))
# for key,values in dc_alph.items():
#     if key == 0:
#         number_entries_dc += math.ceil(values*num_chars)*(1)
#     else:
#         number_entries_dc += math.ceil(values*num_chars)*(key)
    
# print(number_entries_dc/len(dc_compressed))