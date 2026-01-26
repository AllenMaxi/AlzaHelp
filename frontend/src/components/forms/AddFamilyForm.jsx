import React, { useState, useEffect, useCallback, useRef } from 'react';
import { UserPlus, Upload, X, Phone, MapPin, Calendar, FileText, Mic, MicOff, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { familyApi, uploadApi } from '@/services/api';
import { toast } from 'sonner';

const categoryOptions = [
  { value: 'spouse', label: 'Spouse' },
  { value: 'children', label: 'Children' },
  { value: 'grandchildren', label: 'Grandchildren' },
  { value: 'siblings', label: 'Siblings' },
  { value: 'parents', label: 'Parents' },
  { value: 'friends', label: 'Friends' },
  { value: 'other', label: 'Other' },
];

const relationshipOptions = {
  spouse: ['Wife', 'Husband', 'Partner'],
  children: ['Son', 'Daughter', 'Child'],
  grandchildren: ['Grandson', 'Granddaughter', 'Grandchild'],
  siblings: ['Brother', 'Sister', 'Sibling'],
  parents: ['Mother', 'Father', 'Parent'],
  friends: ['Best Friend', 'Close Friend', 'Friend'],
  other: ['Cousin', 'Aunt', 'Uncle', 'Niece', 'Nephew', 'Caregiver', 'Other'],
};

// Voice-to-text hook
const useVoiceToText = () => {
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef(null);
  const onResultRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    setIsSupported(!!SpeechRecognition);

    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            transcript += event.results[i][0].transcript + ' ';
          }
        }
        if (transcript && onResultRef.current) {
          onResultRef.current(transcript);
        }
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        if (event.error === 'not-allowed') {
          toast.error('Please allow microphone access to use voice input');
        }
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {}
      }
    };
  }, []);

  const startListening = useCallback((onResult) => {
    if (recognitionRef.current && !isListening) {
      onResultRef.current = onResult;
      recognitionRef.current.start();
      setIsListening(true);
    }
  }, [isListening]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, [isListening]);

  return { isListening, isSupported, startListening, stopListening };
};

export const AddFamilyForm = ({ onSuccess, onClose }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadingPhoto, setUploadingPhoto] = useState(false);
  const [uploadingVoice, setUploadingVoice] = useState(false);
  
  const { isListening, isSupported, startListening, stopListening } = useVoiceToText();
  
  const [formData, setFormData] = useState({
    name: '',
    category: '',
    relationship: '',
    relationship_label: '',
    phone: '',
    address: '',
    birthday: '',
    photos: [],
    voice_notes: [],
    notes: '',
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleCategoryChange = (value) => {
    setFormData(prev => ({ 
      ...prev, 
      category: value,
      relationship: '',
      relationship_label: ''
    }));
  };

  const handleRelationshipChange = (value) => {
    setFormData(prev => ({ 
      ...prev, 
      relationship: value,
      relationship_label: `Your ${value}`
    }));
  };

  const handlePhotoUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    
    setUploadingPhoto(true);
    try {
      const urls = await Promise.all(files.map(file => uploadApi.uploadFile(file)));
      setFormData(prev => ({ ...prev, photos: [...prev.photos, ...urls] }));
      toast.success('Photo uploaded successfully');
    } catch (error) {
      toast.error('Failed to upload photo');
    } finally {
      setUploadingPhoto(false);
    }
  };

  const handleVoiceUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    
    setUploadingVoice(true);
    try {
      const urls = await Promise.all(files.map(file => uploadApi.uploadFile(file)));
      setFormData(prev => ({ ...prev, voice_notes: [...prev.voice_notes, ...urls] }));
      toast.success('Voice note uploaded successfully');
    } catch (error) {
      toast.error('Failed to upload voice note');
    } finally {
      setUploadingVoice(false);
    }
  };

  const removePhoto = (index) => {
    setFormData(prev => ({
      ...prev,
      photos: prev.photos.filter((_, i) => i !== index)
    }));
  };

  const removeVoiceNote = (index) => {
    setFormData(prev => ({
      ...prev,
      voice_notes: prev.voice_notes.filter((_, i) => i !== index)
    }));
  };

  // Voice input handler for notes
  const handleVoiceInput = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening((transcript) => {
        setFormData(prev => ({
          ...prev,
          notes: prev.notes + transcript
        }));
      });
      toast.info('Listening... speak now to add notes');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.name || !formData.category || !formData.relationship) {
      toast.error('Please fill in the required fields');
      return;
    }
    
    setLoading(true);
    try {
      await familyApi.create(formData);
      toast.success(`${formData.name} has been added to your family!`);
      setFormData({
        name: '',
        category: '',
        relationship: '',
        relationship_label: '',
        phone: '',
        address: '',
        birthday: '',
        photos: [],
        voice_notes: [],
        notes: '',
      });
      setIsOpen(false);
      if (onSuccess) onSuccess();
      if (onClose) onClose();
    } catch (error) {
      toast.error('Failed to add family member');
    } finally {
      setLoading(false);
    }
  };

  const dialogContent = (
    <form onSubmit={handleSubmit} className="space-y-6 mt-4">
      {/* Name */}
      <div>
        <Label htmlFor="name" className="text-lg font-semibold">Name *</Label>
        <Input
          id="name"
          name="name"
          value={formData.name}
          onChange={handleInputChange}
          placeholder="e.g., Maria, Michael, Emma"
          className="mt-2 h-14 text-lg"
          required
        />
      </div>

      {/* Category */}
      <div>
        <Label className="text-lg font-semibold">Relationship Type *</Label>
        <Select value={formData.category} onValueChange={handleCategoryChange}>
          <SelectTrigger className="mt-2 h-14 text-lg">
            <SelectValue placeholder="Select relationship type" />
          </SelectTrigger>
          <SelectContent>
            {categoryOptions.map(opt => (
              <SelectItem key={opt.value} value={opt.value} className="text-lg py-3">
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Relationship */}
      {formData.category && (
        <div>
          <Label className="text-lg font-semibold">Specific Relationship *</Label>
          <Select value={formData.relationship} onValueChange={handleRelationshipChange}>
            <SelectTrigger className="mt-2 h-14 text-lg">
              <SelectValue placeholder="Select relationship" />
            </SelectTrigger>
            <SelectContent>
              {relationshipOptions[formData.category]?.map(rel => (
                <SelectItem key={rel} value={rel} className="text-lg py-3">
                  {rel}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Phone */}
      <div>
        <Label htmlFor="phone" className="text-lg font-semibold flex items-center gap-2">
          <Phone className="h-5 w-5" /> Phone Number
        </Label>
        <Input
          id="phone"
          name="phone"
          value={formData.phone}
          onChange={handleInputChange}
          placeholder="e.g., (555) 123-4567"
          className="mt-2 h-14 text-lg"
        />
      </div>

      {/* Address */}
      <div>
        <Label htmlFor="address" className="text-lg font-semibold flex items-center gap-2">
          <MapPin className="h-5 w-5" /> Address
        </Label>
        <Input
          id="address"
          name="address"
          value={formData.address}
          onChange={handleInputChange}
          placeholder="e.g., 123 Oak Street, Hometown"
          className="mt-2 h-14 text-lg"
        />
      </div>

      {/* Birthday */}
      <div>
        <Label htmlFor="birthday" className="text-lg font-semibold flex items-center gap-2">
          <Calendar className="h-5 w-5" /> Birthday
        </Label>
        <Input
          id="birthday"
          name="birthday"
          value={formData.birthday}
          onChange={handleInputChange}
          placeholder="e.g., March 15 or March 15, 1970"
          className="mt-2 h-14 text-lg"
        />
      </div>

      {/* Photos */}
      <div>
        <Label className="text-lg font-semibold flex items-center gap-2">
          <Upload className="h-5 w-5" /> Photos
        </Label>
        <div className="mt-2">
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handlePhotoUpload}
            className="hidden"
            id="photo-upload"
          />
          <label htmlFor="photo-upload">
            <Button
              type="button"
              variant="outline"
              className="w-full h-14 text-lg cursor-pointer"
              disabled={uploadingPhoto}
              asChild
            >
              <span>
                {uploadingPhoto ? (
                  <><Loader2 className="h-5 w-5 mr-2 animate-spin" /> Uploading...</>
                ) : (
                  <><Upload className="h-5 w-5 mr-2" /> Upload Photos</>
                )}
              </span>
            </Button>
          </label>
        </div>
        {formData.photos.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {formData.photos.map((url, index) => (
              <div key={index} className="relative">
                <img src={url} alt="" className="h-20 w-20 object-cover rounded-lg" />
                <button
                  type="button"
                  onClick={() => removePhoto(index)}
                  className="absolute -top-2 -right-2 bg-destructive text-destructive-foreground rounded-full p-1"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Voice Notes Upload */}
      <div>
        <Label className="text-lg font-semibold flex items-center gap-2">
          <Mic className="h-5 w-5" /> Voice Notes (Upload Recording)
        </Label>
        <div className="mt-2">
          <input
            type="file"
            accept="audio/*"
            multiple
            onChange={handleVoiceUpload}
            className="hidden"
            id="voice-upload"
          />
          <label htmlFor="voice-upload">
            <Button
              type="button"
              variant="outline"
              className="w-full h-14 text-lg cursor-pointer"
              disabled={uploadingVoice}
              asChild
            >
              <span>
                {uploadingVoice ? (
                  <><Loader2 className="h-5 w-5 mr-2 animate-spin" /> Uploading...</>
                ) : (
                  <><Mic className="h-5 w-5 mr-2" /> Upload Voice Notes</>
                )}
              </span>
            </Button>
          </label>
        </div>
        {formData.voice_notes.length > 0 && (
          <div className="space-y-2 mt-3">
            {formData.voice_notes.map((url, index) => (
              <div key={index} className="flex items-center gap-2 p-2 bg-muted rounded-lg">
                <audio src={url} controls className="flex-1 h-10" />
                <button
                  type="button"
                  onClick={() => removeVoiceNote(index)}
                  className="bg-destructive text-destructive-foreground rounded-full p-1"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Notes with Voice-to-Text */}
      <div>
        <Label htmlFor="notes" className="text-lg font-semibold flex items-center gap-2">
          <FileText className="h-5 w-5" /> Special Notes
          {isSupported && (
            <span className="text-sm font-normal text-muted-foreground ml-2">
              (tap mic to speak)
            </span>
          )}
        </Label>
        <div className="mt-2 relative">
          <textarea
            id="notes"
            name="notes"
            value={formData.notes}
            onChange={handleInputChange}
            placeholder="Write anything special about this person - memories, favorite things, etc. You can also tap the microphone to speak."
            className="w-full min-h-[100px] rounded-xl border-2 border-input bg-background px-4 py-3 text-lg focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20 resize-none pr-16"
          />
          {isSupported && (
            <Button
              type="button"
              variant={isListening ? "destructive" : "outline"}
              size="icon"
              onClick={handleVoiceInput}
              className="absolute bottom-3 right-3 h-12 w-12"
              title={isListening ? 'Stop recording' : 'Start voice input'}
            >
              {isListening ? (
                <div className="relative">
                  <MicOff className="h-6 w-6" />
                  <span className="absolute -top-1 -right-1 flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-destructive-foreground opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-destructive-foreground"></span>
                  </span>
                </div>
              ) : (
                <Mic className="h-6 w-6" />
              )}
            </Button>
          )}
        </div>
        {isListening && (
          <p className="text-sm text-primary animate-pulse flex items-center gap-2 mt-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            Listening... speak now to add notes
          </p>
        )}
      </div>

      {/* Submit */}
      <Button
        type="submit"
        variant="accessible"
        className="w-full"
        disabled={loading}
      >
        {loading ? (
          <><Loader2 className="h-5 w-5 mr-2 animate-spin" /> Adding...</>
        ) : (
          <><UserPlus className="h-5 w-5 mr-2" /> Add Family Member</>
        )}
      </Button>
    </form>
  );

  // If used as standalone (with onClose prop), render just the form
  if (onClose) {
    return dialogContent;
  }

  // Otherwise render with dialog trigger
  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="accessible" className="gap-2">
          <UserPlus className="h-5 w-5" />
          Add Family Member
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl font-display flex items-center gap-2">
            <UserPlus className="h-6 w-6 text-primary" />
            Add Family Member
          </DialogTitle>
        </DialogHeader>
        {dialogContent}
      </DialogContent>
    </Dialog>
  );
};

export default AddFamilyForm;
